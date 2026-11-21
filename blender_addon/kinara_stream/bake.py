"""Keyframe baking subsystem for streaming Kinara armatures."""
from array import array
import json
import math
import struct
import time

import bpy

from .constants import MAX_BAKE_FRAMES, OWNER, PoseError, SKELETON
from .rig import forget_rig


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
        self.last_frame = -1
        self.last_kept = -1
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
        enums = bpy.types.Keyframe.bl_rna.properties['interpolation'].enum_items
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
                self.skipped += 1
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
        self.action = bpy.data.actions.new(f'{obj.name} Kinara Motion')
        self.slot = self.action.slots.new('OBJECT', obj.name)
        strip = self.action.layers.new('Motion').strips.new(type='KEYFRAME')
        self.bag = strip.channelbag(self.slot, ensure=True)
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
