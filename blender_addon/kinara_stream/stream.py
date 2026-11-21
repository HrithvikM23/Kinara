"""TCP receiver and stream management for Kinara live capture."""
import json
import socket
import struct
import time
import traceback

import bpy

from .bake import BakeJob
from .constants import (
    ACTOR, ACTOR_FRAME, ACTOR_ID_BYTES, BAKE_TIMEOUT, LIMIT, MAX_ACTORS,
    MAX_BAD_FRAMES, PoseError, RECOVERABLE, SELECTION_COUNT, SNAPSHOT
)
from .rig import canonical_rigs, RigTarget, valid_actor_id


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
        sock = socket.create_connection(('127.0.0.1', port), timeout=2.0)
        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.setblocking(False)
        except OSError:
            sock.close()
            raise
        self.sock = sock
        if context is not None:
            self.scene_id, self.target_layer = context.scene.session_uid, context.view_layer.name
        self.status = 'Connected - waiting for people'

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
        self.bake_status = 'Requesting recorded motion...'

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
                self.bake_status = f'Receiving motion - {len(job.times)} recorded frames{skipped}'
            elif kind == ord('Z'):
                job.prepare_action()
                self.bake_status = 'Writing Blender keyframes...'
            else:
                self.stop_bake(payload.decode('utf-8', errors='replace'))
        except RECOVERABLE as error:
            self.stop_bake(str(error))

    def advance_bake(self, context):
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
        message = f'Baked {len(job.times)} frames{skipped} - rig disconnected'
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
                bad = True
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
        self.status = f'Connected - {count} people' if count else 'Connected - waiting for people'
        if bad:
            self.status += ' - skipping invalid pose'
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
                if len(self.data) < LIMIT + 4:
                    chunk = self.sock.recv(256 * 1024)
                    if not chunk:
                        self.close('Stream stopped in Kinara')
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
                    schema['actor'] = schema['actor'].lower()
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
                    self.status = 'Paused - return to the connected scene and view layer'
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
            traceback.print_exc()
            self.close(str(error))
            return True


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
        if (receiver.selected is None or receiver.visible is None
                or not receiver.in_target(context)):
            return None
        selected, visible = receiver.selected, receiver.visible
    else:
        selected, visible = (), ()
    candidates = {obj for obj in canonical_rigs(context.scene)
                  if obj.get(ACTOR) not in selected and obj not in visible
                  and obj.library is None and len(obj.users_scene) == 1}
    users = bpy.data.user_map(subset=candidates) if candidates else {}
    return sorted((obj for obj in candidates if obj.override_library is None
                   and all(isinstance(user, (bpy.types.Collection, bpy.types.Scene))
                           for user in users.get(obj, ()))), key=lambda obj: obj.name)
