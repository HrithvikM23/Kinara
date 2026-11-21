"""Kinara Motion Feed Streamer for Blender."""
# pyright: reportIncompatibleMethodOverride=false, reportOptionalMemberAccess=false

bl_info = {
    'name': 'Kinara Live Stream',
    'blender': (4, 0, 0),
    'version': (1, 4, 0),
    'category': 'Animation',
    'author': 'Kinara Core Team',
    'description': 'Receives a live markerless bone stream from Kinara and bakes it to keyframes.',
}

import bpy

from .constants import PORT
from .stream import receiver, stop_timer, tick
from .ui import (
    after_undo, before_load, CLASSES
)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.WindowManager.kinara_bone_stream_port = bpy.props.IntProperty(
        name='Port', default=PORT, min=1024, max=65535, options={'SKIP_SAVE'}
    )
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
    if hasattr(bpy.types.WindowManager, 'kinara_bone_stream_port'):
        del bpy.types.WindowManager.kinara_bone_stream_port
    for cls in reversed(CLASSES):
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            pass


__all__ = ['bl_info', 'register', 'unregister']
