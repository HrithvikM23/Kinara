"""Rig target construction and hierarchy binding for Kinara."""
from hashlib import sha256
import json
import math
import string
import struct

import bpy
from mathutils import Matrix

from .constants import (
    ACTOR, BONE_INDEX, COLLECTION, OWNER, PoseError, REGISTRY_PREFIX, SKELETON
)


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


def matching_hierarchy(bones, parents):
    return all(bone is not None for bone in bones) and all(
        bone.parent == (bones[parent] if parent >= 0 else None)
        for bone, parent in zip(bones, parents))


def matrix(values):
    return Matrix((values[:4], values[4:8], values[8:12], (0, 0, 0, 1)))


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
                    any(abs(axes.col[j].dot(axes.col[k])-(j == k)) > 0.001
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
                raise ValueError('Several existing armatures share this stream ID. Keep one before reconnecting.')
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
        bases = self.basis(payload, offset)
        for name, basis in zip(self.names, bases):
            self.object.pose.bones[name].matrix_basis = basis
