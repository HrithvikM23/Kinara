"""Kinara feed streamer for Blender."""
# pyright: reportIncompatibleMethodOverride=false, reportOptionalMemberAccess=false

bl_info = {
    'name': 'Kinara', 'blender': (4, 5, 0),
    'version': (1, 4, 0), 'category': 'Animation',
    'description': 'Receives a live bone stream and bakes it to keyframes.',
}

from hashlib import sha256
from array import array
import json
import math
import socket
import string
import struct
import time
import traceback

import bpy
from bpy.app.handlers import persistent
from mathutils import Matrix

PORT = 49168
LIMIT = 1024 * 1024
SNAPSHOT = struct.Struct('<QH')
ACTOR_FRAME = struct.Struct('<II')
SELECTION_COUNT = struct.Struct('<H')
ACTOR_ID_BYTES = 32
MAX_ACTORS = 128
MAX_BAKE_FRAMES = 36000
MAX_BAD_FRAMES = 60          # consecutive snapshots with an unusable pose before disconnecting
BAKE_TIMEOUT = 15.0          # seconds without bake packets before a pending bake is abandoned
COLLECTION = 'RGB MoCap Stream'
OWNER = 'rgbmocap_bone_stream'
SKELETON = 'rgbmocap_stream_skeleton'
ACTOR = 'rgbmocap_stream_actor'
BONE_INDEX = 'rgbmocap_stream_bone_index'
REGISTRY_PREFIX = 'stream_rig_'

# Errors that mean "this connection or job is bad", not "the add-on has a bug".
RECOVERABLE = (OSError, ValueError, TypeError, KeyError, RuntimeError, ReferenceError, struct.error)


class PoseError(ValueError):
    """A single pose frame is unusable. The stream itself is still fine."""


def valid_actor_id(value):
    return (isinstance(value, str) and len(value) == 64
            and all(c in string.hexdigits for c in value))


def managed_collection(scene):
    return next((c for c in scene.collection.children if c.get(OWNER)), None)


def registry_key(actor, skeleton):
    return REGISTRY_PREFIX + sha256(f'{actor}/{skeleton}'.encode('utf-8')).hexdigest()[:48]


def tagged_rigs(scene):
    return [obj for obj in scene.objects if obj.type == 'ARMATURE' and obj.get(OWNER)
            and obj.get(ACTOR) and obj.get(SKELETON)]


def canonical_rigs(scene):
    collection = managed_collection(scene)
    by_identity = {}
    for obj in tagged_rigs(scene):
        key = registry_key(obj[ACTOR], obj[SKELETON])
        by_identity.setdefault(key, []).append(obj)
    for key, matches in by_identity.items():
        registered = collection.get(key) if collection is not None else None
        if registered is not None and registered in matches:
            yield registered
        elif len(matches) == 1 and (collection is None or key not in collection):
            yield matches[0]


def forget_rig(scene, obj):
    collection = managed_collection(scene)
    if collection is not None:
        key = registry_key(obj[ACTOR], obj[SKELETON])
        if collection.get(key) == obj:
            collection[key] = False


class RigTarget:
    def __init__(self, context, schema):
        bones = schema['bones']
        if not 1 <= len(bones) <= 128:
            raise ValueError('Unsupported skeleton description')
        names, parents, binds = [], [], []
        for index, bone in enumerate(bones):
            name, parent, rows, length = bone
            if (not isinstance(name, str) or not 1 <= len(name.encode('utf-8')) <= 63
                    or '\0' in name or name in names or type(parent) is not int
                    or (parent != -1 if index == 0 else not 0 <= parent < index)
                    or not 0.0001 <= length <= 10 or len(rows) != 12
                    or not all(math.isfinite(x) for x in rows)):
                raise ValueError('Invalid bone description')
            bind = matrix(rows)
            axes = bind.to_3x3()
            if (abs(axes.determinant()-1) > 0.001 or
                    any(abs(axes.col[j].dot(axes.col[k])-(j == k)) > 0.001  # type: ignore
                        for j in range(3) for k in range(3))):
                raise ValueError('Invalid rest rotation')
            names.append(name)
            parents.append(parent)
            binds.append((bind, length))

        key = sha256(json.dumps(bones, ensure_ascii=False, separators=(',', ':')).encode('utf-8')).hexdigest()
        actor = schema['actor']
        registry = registry_key(actor, key)
        collection = managed_collection(context.scene)
        obj = next((obj for obj in canonical_rigs(context.scene)
                    if obj[ACTOR] == actor and obj[SKELETON] == key), None)
        if obj is not None:
            if context.view_layer.objects.get(obj.name) is None:
                raise ValueError('The stream armature is excluded from this view layer.')
        else:
            if (collection is None or registry not in collection) and any(
                    rig[ACTOR] == actor and rig[SKELETON] == key for rig in tagged_rigs(context.scene)):
                raise ValueError('Several legacy armatures share this stream ID. Keep one before reconnecting.')
            obj = self.create_rig(context, collection, names, parents, binds, schema['name'])
            obj[OWNER], obj[ACTOR], obj[SKELETON] = True, actor, key
            created = obj.data
            if not isinstance(created, bpy.types.Armature):
                raise ValueError('Stream object is not an armature.')
            for index, name in enumerate(names):
                created.bones[name][BONE_INDEX] = index

        armature = obj.data
        if not isinstance(armature, bpy.types.Armature):
            raise ValueError('Stream object is not an armature.')
        indexed = {bone[BONE_INDEX]: bone for bone in armature.bones if BONE_INDEX in bone}
        ordered = [indexed.get(i) for i in range(len(names))]
        if (len(indexed) != sum(BONE_INDEX in bone for bone in armature.bones)
                or not matching_hierarchy(ordered, parents)):
            raise ValueError('Stream armature hierarchy was edited. Restore its original parent links to reconnect.')
        found = [bone for bone in ordered if bone is not None]
        actual = [bone.matrix_local.copy() for bone in found]
        self.object, self.names, self.parents = obj, [bone.name for bone in found], parents
        self.bind_inverse = [
            (actual[parent].inverted() @ actual[index] if parent >= 0 else actual[index]).inverted()
            for index, parent in enumerate(parents)]
        collection = managed_collection(context.scene)
        if collection is not None:
            collection[registry] = obj

    @staticmethod
    def create_rig(context, collection, names, parents, binds, actor_name):
        if collection is None:
            collection = bpy.data.collections.new(COLLECTION)
            collection[OWNER] = True
            context.scene.collection.children.link(collection)
        armature = bpy.data.armatures.new(actor_name)
        obj = bpy.data.objects.new(actor_name, armature)
        object_name = obj.name
        collection.objects.link(obj)
        obj.show_in_front = True
        armature.display_type = 'OCTAHEDRAL'
        previous = context.view_layer.objects.active
        try:
            if context.view_layer.objects.get(obj.name) is None:
                raise ValueError('The stream collection is excluded from this view layer.')
            obj.select_set(True)
            context.view_layer.objects.active = obj
            with context.temp_override(object=obj, active_object=obj,
                                       selected_objects=[obj], selected_editable_objects=[obj]):
                bpy.ops.object.mode_set(mode='EDIT')
                try:
                    for index, (bind, length) in enumerate(binds):
                        edit = armature.edit_bones.new(names[index])
                        edit.head = bind.translation
                        edit.tail = bind.translation + bind.to_3x3().col[1] * length
                        edit.align_roll(bind.to_3x3().col[2])
                        if parents[index] >= 0:
                            edit.parent = armature.edit_bones[names[parents[index]]]
                finally:
                    bpy.ops.object.mode_set(mode='OBJECT')
        except Exception:
            bpy.data.objects.remove(obj, do_unlink=True)
            bpy.data.armatures.remove(armature)
            raise
        finally:
            context.view_layer.objects.active = previous
            if object_name in bpy.data.objects:
                obj.select_set(False)
        return obj

    def basis(self, payload, offset):
        """Return the pose-bone basis matrices for one frame.

        Raises PoseError (never a partial result) if the frame is unusable.
        """
        values = struct.unpack_from(f'<{len(self.names)*12}f', payload, offset)
        if not all(math.isfinite(value) for value in values):
            raise PoseError('Invalid pose numbers.')
        matrices = [matrix(values[i:i+12]) for i in range(0, len(values), 12)]
        try:
            inverses = [pose.inverted() for pose in matrices]
        except ValueError as error:
            raise PoseError('Singular pose matrix.') from error
        result = []
        for i, parent in enumerate(self.parents):
            relative = inverses[parent] @ matrices[i] if parent >= 0 else matrices[i]
            result.append(self.bind_inverse[i] @ relative)
        return result

    def apply(self, payload, offset):
        bases = self.basis(payload, offset)  # compute everything first: no half-applied poses
        for name, basis in zip(self.names, bases):
            self.object.pose.bones[name].matrix_basis = basis


class BakeJob:
    def __init__(self, context, rig, schema):
        self.scene = context.scene
        self.layer = context.view_layer
        self.rig = rig
        self.actor = schema['actor']
        self.key = rig.object[SKELETON]
        self.fields = []
        self.tracks = []
        self.previous = [None] * len(rig.names)
        for name in rig.names:
            bone = rig.object.pose.bones[name]
            mode = bone.rotation_mode
            prop = {'QUATERNION': 'rotation_quaternion',
                    'AXIS_ANGLE': 'rotation_axis_angle'}.get(mode, 'rotation_euler')
            channels = [('location', 3), (prop, 3 if prop == 'rotation_euler' else 4), ('scale', 3)]
            fields = [(bone.path_from_id(path), index) for path, size in channels for index in range(size)]
            self.fields.append((mode, fields))
            self.tracks.append([array('f') for _ in fields])
        self.times = array('f')
        self.interpolation = array('i')
        self.last_frame = -1   # last frame number received (monotonic check)
        self.last_kept = -1    # last frame number actually recorded (gap detection)
        self.skipped = 0
        self.count = None
        self.rate = 1.0
        self.linear = 0
        self.constant = 0
        self.action = None
        self.slot = None
        self.bag = None
        self.curves = iter(())
        self.writing = False

    def start(self, payload):
        if self.count is not None:
            raise ValueError('Duplicate bake header.')
        meta = json.loads(payload)
        fps, count = float(meta['fps']), int(meta['frames'])
        if (str(meta['actor']).lower() != self.actor or meta['rig'] != self.key or
                not math.isfinite(fps) or not 0 < fps <= 240 or not 1 <= count <= MAX_BAKE_FRAMES):
            raise ValueError('Unexpected bake source.')
        self.count = count
        self.rate = self.scene.render.fps / (self.scene.render.fps_base * fps)
        enums = bpy.types.Keyframe.bl_rna.properties['interpolation'].enum_items  # type: ignore
        self.linear = enums['LINEAR'].value
        self.constant = enums['CONSTANT'].value

    def receive(self, payload):
        size = 4 + len(self.rig.names) * 48
        if self.count is None or not payload or len(payload) % size:
            raise ValueError('Invalid bake frame data.')
        for offset in range(0, len(payload), size):
            frame, = struct.unpack_from('<I', payload, offset)
            if not self.last_frame < frame < self.count:
                raise ValueError('Unexpected bake frame number.')
            self.last_frame = frame
            try:
                bases = self.rig.basis(payload, offset+4)
            except PoseError:
                self.skipped += 1   # drop the frame; the gap is held by constant interpolation
                continue
            if self.times and frame > self.last_kept + 1:
                self.interpolation[-1] = self.constant
            self.last_kept = frame
            self.times.append(1 + frame * self.rate)
            self.interpolation.append(self.linear)
            for i, basis in enumerate(bases):
                location, rotation, scale = basis.decompose()
                mode, _ = self.fields[i]
                previous = self.previous[i]
                if mode in ('QUATERNION', 'AXIS_ANGLE'):
                    if previous is not None and previous.dot(rotation) < 0:
                        rotation.negate()
                    self.previous[i] = rotation
                    values = tuple(rotation) if mode == 'QUATERNION' else (rotation.angle, *rotation.axis)
                else:
                    euler = rotation.to_euler(mode, previous) if previous is not None else rotation.to_euler(mode)
                    self.previous[i] = euler
                    values = tuple(euler)
                for track, value in zip(self.tracks[i], (*location, *values, *scale)):
                    track.append(value)

    def prepare_action(self):
        if not self.times:
            raise ValueError('There are no recorded frames to bake.')
        obj = self.rig.object
        self.action = bpy.data.actions.new(f'{obj.name} RGB MoCap')
        self.slot = self.action.slots.new('OBJECT', obj.name)
        strip = self.action.layers.new('Motion').strips.new(type='KEYFRAME')
        self.bag = strip.channelbag(self.slot, ensure=True)  # type: ignore
        self.curves = iter([(path, component, values)
                            for (_, fields), tracks in zip(self.fields, self.tracks)
                            for (path, component), values in zip(fields, tracks)])
        self.writing = True

    def write_curves(self, until):
        while time.perf_counter() < until:
            curve_data = next(self.curves, None)
            if curve_data is None:
                return True
            path, component, values = curve_data
            curve = self.bag.fcurves.new(path, index=component)
            curve.keyframe_points.add(len(self.times))
            coords = array('f', [0]) * (2 * len(self.times))
            coords[::2], coords[1::2] = self.times, values
            curve.keyframe_points.foreach_set('co', coords)
            curve.keyframe_points.foreach_set('interpolation', self.interpolation)
            curve.update()
        return False

    def check_context(self, context):
        obj = self.rig.object
        if (context.scene != self.scene or context.view_layer != self.layer or context.mode != 'OBJECT'
                or context.view_layer.objects.get(obj.name) != obj):
            raise ValueError('Bake stopped because the Blender scene or mode changed.')

    def commit(self):
        obj = self.rig.object
        if not obj.get(OWNER) or any(
                obj.pose.bones.get(name) is None or obj.pose.bones[name].rotation_mode != mode
                for name, (mode, _) in zip(self.rig.names, self.fields)):
            raise ValueError('The rig changed during baking.')
        animation = obj.animation_data_create()
        if animation.use_tweak_mode:
            raise ValueError('Exit NLA Tweak Mode before baking.')
        previous, slot = animation.action, animation.action_slot
        try:
            animation.action = self.action
            animation.action_slot = self.slot
        except (TypeError, RuntimeError):
            animation.action = previous
            if previous is not None:
                animation.action_slot = slot
            raise
        if previous is not None:
            previous.use_fake_user = True
        self.action = None
        forget_rig(self.scene, obj)
        del obj[OWNER]
        obj.hide_set(False)

    def cancel(self):
        if self.action is not None:
            bpy.data.actions.remove(self.action)
            self.action = None


class Receiver:
    def __init__(self):
        self.sock = None
        self.data = bytearray()
        self.schemas = {}
        self.rigs = {}
        self.sequence = -1
        self.snapshot = None
        self.view_layer = None
        self.visible = None
        self.selected = None
        self.bake = None
        self.bake_pending = False
        self.bake_activity = 0.0
        self.bake_status = ''
        self.detached = set()
        self.scene_id = self.target_layer = None
        self.generation = 0
        self.selection_requested = False
        self.paused_for_context = False
        self.bad_frames = 0
        self.status = 'Disconnected'

    def send(self, data):
        sock = self.sock
        if sock is None:
            raise OSError('Stream disconnected.')
        sock.sendall(data)

    def connect(self, port, context=None):
        self.close()
        sock = socket.create_connection(('127.0.0.1', port), timeout=.15)
        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.setblocking(False)
        except OSError:
            sock.close()
            raise
        self.sock = sock
        if context is not None:
            self.scene_id, self.target_layer = context.scene.session_uid, context.view_layer.name
        self.status = 'Connected · waiting for people'

    def close(self, status='Disconnected'):
        self.generation += 1
        self.bake_status = ''
        if self.bake is not None:
            self.stop_bake('Bake stopped because the stream disconnected.')
        self.bake_pending = False
        self.detached.clear()
        if self.sock is not None:
            self.sock.close()
        self.sock = None
        self.data.clear()
        self.schemas.clear()
        self.rigs.clear()
        self.sequence = -1
        self.snapshot = None
        self.view_layer = None
        self.visible = None
        self.selected = None
        self.scene_id = self.target_layer = None
        self.selection_requested = False
        self.paused_for_context = False
        self.bad_frames = 0
        self.status = status

    def in_target(self, context):
        return self.scene_id == context.scene.session_uid and self.target_layer == context.view_layer.name

    def bake_target(self, context):
        if (self.sock is None or self.bake_pending or self.bake is not None
                or context.mode != 'OBJECT' or not self.in_target(context)
                or self.view_layer != context.view_layer):
            return None
        targets = [(rig, self.schemas[slot]) for slot, rig in self.rigs.items()
                   if self.schemas[slot].get('bake') == 1]
        return next((pair for pair in targets if pair[0].object == context.active_object),
                    targets[0] if len(targets) == 1 else None)

    def request_bake(self, context):
        target = self.bake_target(context)
        if target is None:
            raise ValueError('Select an actively streaming armature in Object Mode.')
        job = BakeJob(context, *target)
        self.send(b'B' + bytes.fromhex(job.actor) + bytes.fromhex(job.key))
        self.bake = job
        self.bake_pending = True
        self.bake_activity = time.monotonic()
        self.bake_status = 'Requesting recorded motion…'

    def stop_bake(self, message):
        if self.bake is not None:
            self.bake.cancel()
        self.bake = None
        self.bake_status = message

    def bake_packet(self, kind, payload):
        job = self.bake
        if kind in (ord('F'), ord('Z')):
            self.bake_pending = False
        self.bake_activity = time.monotonic()
        if job is None:
            return
        try:
            if kind == ord('H'):
                job.start(payload)
            elif kind == ord('D'):
                job.receive(payload)
                skipped = f' ({job.skipped} skipped)' if job.skipped else ''
                self.bake_status = f'Receiving motion · {len(job.times)} recorded frames{skipped}'
            elif kind == ord('Z'):
                job.prepare_action()
                self.bake_status = 'Writing Blender keyframes…'
            else:
                self.stop_bake(payload.decode('utf-8', errors='replace'))
        except RECOVERABLE as error:
            self.stop_bake(str(error))

    def advance_bake(self, context):
        # A pending bake that goes quiet is abandoned, which re-enables the Bake button.
        if self.bake_pending and time.monotonic() - self.bake_activity > BAKE_TIMEOUT:
            self.bake_pending = False
            if self.bake is not None:
                self.stop_bake('The app stopped sending bake frames. Try the bake again.')
            return True
        job = self.bake
        if job is None:
            return False
        try:
            job.check_context(context)
            if not job.writing:
                return False
            if not job.write_curves(time.perf_counter() + .008):
                return True
            job.commit()
        except RECOVERABLE as error:
            self.stop_bake(str(error))
            return True
        self.detached.add(job.actor)
        self.rigs = {slot: rig for slot, rig in self.rigs.items() if self.schemas[slot]['actor'] != job.actor}
        if self.visible is not None:
            self.visible.discard(job.rig.object)
        self.bake = None
        skipped = f', {job.skipped} skipped' if job.skipped else ''
        message = f'Baked {len(job.times)} frames{skipped} · rig disconnected'
        self.bake_status = message
        try:
            self.send(b'A' + bytes.fromhex(job.actor))
        except OSError:
            self.close('Stream stopped after baking.')
            self.bake_status = message
        return True

    def apply(self, context, payload):
        sequence, count = SNAPSHOT.unpack_from(payload)
        if sequence <= self.sequence:
            return False
        if count > MAX_ACTORS:
            raise ValueError('Too many people in stream.')
        offset = SNAPSHOT.size
        entries = {}
        for _ in range(count):
            slot, _frame = ACTOR_FRAME.unpack_from(payload, offset)
            schema = self.schemas[slot]
            offset += ACTOR_FRAME.size
            end = offset + len(schema['bones']) * 48
            if slot in entries or end > len(payload):
                raise ValueError('Invalid pose size or slot.')
            entries[slot] = offset
            offset = end
        selected = None
        if offset < len(payload):
            total, = SELECTION_COUNT.unpack_from(payload, offset)
            offset += SELECTION_COUNT.size
            if total > MAX_ACTORS or len(payload) != offset + total * ACTOR_ID_BYTES:
                raise ValueError('Invalid stream selection.')
            selected = {payload[start:start + ACTOR_ID_BYTES].hex()
                        for start in range(offset, len(payload), ACTOR_ID_BYTES)}

        if selected is not None:
            self.detached.intersection_update(selected)
        visible = set()
        live = {}
        bad = False
        for slot, offset in entries.items():
            if self.schemas[slot]['actor'] in self.detached:
                continue
            rig = self.rigs.get(slot)
            if rig is None:
                rig = RigTarget(context, self.schemas[slot])
            try:
                rig.apply(payload, offset)
            except PoseError:
                bad = True  # keep the previous pose for this person and carry on
            live[slot] = rig
            visible.add(rig.object)
        self.bad_frames = self.bad_frames + 1 if bad else 0
        if self.bad_frames > MAX_BAD_FRAMES:
            raise ValueError('Too many invalid pose frames in a row.')
        self.rigs = live
        if self.visible != visible:
            for obj in canonical_rigs(context.scene):
                if context.view_layer.objects.get(obj.name) == obj and obj.hide_get() == (obj in visible):
                    obj.hide_set(obj not in visible)
            self.visible = visible
        self.sequence = sequence
        self.selected = selected
        self.status = f'Connected · {count} people' if count else 'Connected · waiting for people'
        if bad:
            self.status += ' · skipping invalid pose'
        return True

    def tick(self, context):
        if self.sock is None:
            return False
        try:
            if self.scene_id is None:
                self.scene_id, self.target_layer = context.scene.session_uid, context.view_layer.name
            redraw = self.advance_bake(context)
            if self.sock is None:
                return True
            try:
                # +4 so a maximum-size message (LIMIT bytes + 4-byte header) can always complete.
                if len(self.data) < LIMIT + 4:
                    chunk = self.sock.recv(256 * 1024)
                    if not chunk:
                        self.close('Stream stopped in RGB MoCap')
                        return True
                    self.data.extend(chunk)
            except BlockingIOError:
                pass
            stop = time.perf_counter() + .004
            while len(self.data) >= 5 and time.perf_counter() < stop:
                size, = struct.unpack_from('<I', self.data)
                if not 2 <= size <= LIMIT:
                    raise ValueError('Invalid stream message size.')
                if len(self.data) < size+4:
                    break
                kind = self.data[4]
                payload = bytes(self.data[5:size+4])
                del self.data[:size+4]
                if kind == ord('R'):
                    schema = json.loads(payload)
                    slot = schema['slot']
                    if (schema['version'] != 2 or type(slot) is not int or not 1 <= slot <= MAX_ACTORS
                            or not valid_actor_id(schema['actor'])
                            or not isinstance(schema['name'], str) or not 1 <= len(schema['bones']) <= 128):
                        raise ValueError('Invalid stream rig declaration.')
                    schema['actor'] = schema['actor'].lower()  # selection IDs are compared as lowercase hex
                    self.schemas[slot] = schema
                    if schema.get('selection') == 1 and not self.selection_requested:
                        self.send(b'V')
                        self.selection_requested = True
                    self.rigs.pop(slot, None)
                    self.snapshot = None
                elif kind == ord('S'):
                    self.snapshot = payload
                elif kind in (ord('H'), ord('D'), ord('Z'), ord('F')):
                    self.bake_packet(kind, payload)
                    redraw = True
                else:
                    raise ValueError('Unexpected stream message.')
            if not self.in_target(context):
                if not self.paused_for_context:
                    self.paused_for_context = True
                    self.status = 'Paused · return to the connected scene and view layer'
                    return True
                return redraw
            if self.paused_for_context:
                self.paused_for_context = False
                self.sequence = -1
            if context.mode != 'OBJECT':
                self.rigs.clear()
                self.sequence = -1
                return False
            if context.view_layer != self.view_layer:
                self.view_layer = context.view_layer
                self.rigs.clear()
                self.sequence = -1
                self.visible = None
            updated = self.snapshot is not None and self.apply(context, self.snapshot)
            return redraw or updated
        except RECOVERABLE as error:
            traceback.print_exc()  # full detail in Blender's system console
            self.close(str(error))
            return True


def matching_hierarchy(bones, parents):
    return all(bone is not None for bone in bones) and all(
        bone.parent == (bones[parent] if parent >= 0 else None)
        for bone, parent in zip(bones, parents))


def matrix(values):
    return Matrix((values[:4], values[4:8], values[8:12], (0, 0, 0, 1)))


receiver = Receiver()


def tick():
    if receiver.tick(bpy.context) and bpy.context.screen:
        for area in bpy.context.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()
    return 0.02 if receiver.sock is not None else None


def stop_timer():
    if bpy.app.timers.is_registered(tick):
        bpy.app.timers.unregister(tick)


def inactive_rigs(context):
    if receiver.sock is not None:
        # A missing pose can be a gap in a checked person's take.
        if (receiver.selected is None or receiver.visible is None
                or not receiver.in_target(context)):
            return None
        selected, visible = receiver.selected, receiver.visible
    else:
        selected, visible = (), ()
    candidates = {obj for obj in canonical_rigs(context.scene)
                  if obj.get(ACTOR) not in selected and obj not in visible
                  and obj.library is None and len(obj.users_scene) == 1}
    users = bpy.data.user_map(subset=candidates) if candidates else {}  # type: ignore
    return sorted((obj for obj in candidates if obj.override_library is None
                   and all(isinstance(user, (bpy.types.Collection, bpy.types.Scene))
                           for user in users.get(obj, ()))), key=lambda obj: obj.name)


@persistent
def before_load(_):
    stop_timer()
    receiver.close()


@persistent
def after_undo(_):
    receiver.generation += 1
    receiver.rigs.clear()
    receiver.view_layer = None
    receiver.selected = None


class RGBMOCAP_OT_BoneStreamConnect(bpy.types.Operator):
    bl_idname = 'rgbmocap_bone_stream.connect'
    bl_label = 'Connect'

    def execute(self, context):
        try:
            receiver.connect(context.window_manager.rgbmocap_bone_stream_port, context)  # type: ignore
        except OSError as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}
        if not bpy.app.timers.is_registered(tick):
            bpy.app.timers.register(tick)
        return {'FINISHED'}


class RGBMOCAP_OT_BoneStreamDisconnect(bpy.types.Operator):
    bl_idname = 'rgbmocap_bone_stream.disconnect'
    bl_label = 'Disconnect'

    def execute(self, context):
        stop_timer()
        receiver.close()
        return {'FINISHED'}


class RGBMOCAP_OT_BoneStreamClean(bpy.types.Operator):
    bl_idname = 'rgbmocap_bone_stream.clean'
    bl_label = 'Clean'
    bl_description = 'Remove RGB MoCap person rigs outside the current stream'
    bl_options = {'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT'

    def candidates(self, context):
        rigs = inactive_rigs(context)
        if rigs is None:
            self.report({'WARNING'}, 'Waiting for the current stream selection. Restart RGB MoCap if needed.')
        elif not rigs:
            self.report({'INFO'}, 'No inactive RGB MoCap rigs to clean.')
        return rigs

    def invoke(self, context, event):
        rigs = self.candidates(context)
        if not rigs:
            return {'CANCELLED'}
        self.confirmed_rigs = {obj.session_uid for obj in rigs}
        self.confirmed_context = (receiver.generation, context.scene.session_uid, context.view_layer.name)
        message = (f'Remove {len(rigs)} inactive RGB MoCap rig(s) from this scene?' if receiver.sock else
                   f'Stream is disconnected. Remove all {len(rigs)} eligible RGB MoCap rig(s) from this scene?')
        return context.window_manager.invoke_confirm(self, event, title='Clean RGB MoCap Rigs',
                                                     message=message, confirm_text='Remove')

    def execute(self, context):
        confirmation = getattr(self, 'confirmed_context', None)
        if context.mode != 'OBJECT' or (confirmation is not None and confirmation != (
                receiver.generation, context.scene.session_uid, context.view_layer.name)):
            self.report({'WARNING'}, 'Connection or scene changed. Open Clean again.')
            return {'CANCELLED'}
        rigs = self.candidates(context)
        if not rigs:
            return {'CANCELLED'}
        confirmed = getattr(self, 'confirmed_rigs', None)
        removed = 0
        for obj in rigs:
            if confirmed is not None and obj.session_uid not in confirmed:
                continue
            armature = obj.data
            forget_rig(context.scene, obj)
            bpy.data.objects.remove(obj, do_unlink=True)
            if armature.users == 0 and armature.library is None:
                bpy.data.armatures.remove(armature)
            removed += 1
        self.report({'INFO'}, f'Removed {removed} inactive RGB MoCap rig(s).')
        return {'FINISHED'} if removed else {'CANCELLED'}


class RGBMOCAP_OT_BoneStreamBake(bpy.types.Operator):
    bl_idname = 'rgbmocap_bone_stream.bake'
    bl_label = 'Bake'
    bl_description = 'Bake all recorded motion to the selected streaming rig and detach it'

    @classmethod
    def poll(cls, context):
        return receiver.bake_target(context) is not None

    def execute(self, context):
        try:
            receiver.request_bake(context)
        except RECOVERABLE as error:
            self.report({'WARNING'}, str(error))
            return {'CANCELLED'}
        return {'FINISHED'}


class RGBMOCAP_PT_BoneStream(bpy.types.Panel):
    bl_label = 'RGB MoCap Bone Stream'
    bl_idname = 'RGBMOCAP_PT_bone_stream'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'RGB MoCap'

    def draw(self, context):
        layout = self.layout
        assert layout is not None
        layout.prop(context.window_manager, 'rgbmocap_bone_stream_port')
        layout.operator('rgbmocap_bone_stream.disconnect' if receiver.sock else 'rgbmocap_bone_stream.connect')
        layout.operator('rgbmocap_bone_stream.bake')
        clean_row = layout.row()
        clean_row.operator_context = 'INVOKE_DEFAULT'
        clean_row.operator('rgbmocap_bone_stream.clean')
        layout.label(text=receiver.status)
        if receiver.bake_status:
            layout.label(text=receiver.bake_status)


CLASSES = (RGBMOCAP_OT_BoneStreamConnect, RGBMOCAP_OT_BoneStreamDisconnect,
           RGBMOCAP_OT_BoneStreamClean, RGBMOCAP_OT_BoneStreamBake, RGBMOCAP_PT_BoneStream)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.WindowManager.rgbmocap_bone_stream_port = bpy.props.IntProperty(  # type: ignore
        name='Port', default=PORT, min=1024, max=65535, options={'SKIP_SAVE'})
    bpy.app.handlers.load_pre.append(before_load)
    bpy.app.handlers.undo_post.append(after_undo)
    bpy.app.handlers.redo_post.append(after_undo)


def unregister():
    before_load(None)
    for handlers, fn in ((bpy.app.handlers.load_pre, before_load),
                         (bpy.app.handlers.undo_post, after_undo),
                         (bpy.app.handlers.redo_post, after_undo)):
        if fn in handlers:
            handlers.remove(fn)
    if hasattr(bpy.types.WindowManager, 'rgbmocap_bone_stream_port'):
        del bpy.types.WindowManager.rgbmocap_bone_stream_port  # type: ignore
    for cls in reversed(CLASSES):
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            pass