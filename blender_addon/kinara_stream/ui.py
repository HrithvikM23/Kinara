"""User interface, panels, and operators for Kinara Blender streaming."""
import bpy
from bpy.app.handlers import persistent

from .constants import PORT
from .rig import forget_rig
from .stream import inactive_rigs, receiver, stop_timer, tick


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


class KINARA_OT_StreamConnect(bpy.types.Operator):
    bl_idname = 'kinara_bone_stream.connect'
    bl_label = 'Connect'

    def execute(self, context):
        try:
            receiver.connect(context.window_manager.kinara_bone_stream_port, context)
        except OSError as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}
        if not bpy.app.timers.is_registered(tick):
            bpy.app.timers.register(tick)
        return {'FINISHED'}


class KINARA_OT_StreamDisconnect(bpy.types.Operator):
    bl_idname = 'kinara_bone_stream.disconnect'
    bl_label = 'Disconnect'

    def execute(self, context):
        stop_timer()
        receiver.close()
        return {'FINISHED'}


class KINARA_OT_StreamClean(bpy.types.Operator):
    bl_idname = 'kinara_bone_stream.clean'
    bl_label = 'Clean'
    bl_description = 'Remove Kinara person rigs outside the current stream'
    bl_options = {'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT'

    def candidates(self, context):
        rigs = inactive_rigs(context)
        if rigs is None:
            self.report({'WARNING'}, 'Waiting for the current stream selection. Restart Kinara if needed.')
        elif not rigs:
            self.report({'INFO'}, 'No inactive Kinara rigs to clean.')
        return rigs

    def invoke(self, context, event):
        rigs = self.candidates(context)
        if not rigs:
            return {'CANCELLED'}
        self.confirmed_rigs = {obj.session_uid for obj in rigs}
        self.confirmed_context = (receiver.generation, context.scene.session_uid, context.view_layer.name)
        message = (f'Remove {len(rigs)} inactive Kinara rig(s) from this scene?' if receiver.sock else
                   f'Stream is disconnected. Remove all {len(rigs)} eligible Kinara rig(s) from this scene?')
        return context.window_manager.invoke_confirm(self, event, title='Clean Kinara Rigs',
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
        self.report({'INFO'}, f'Removed {removed} inactive Kinara rig(s).')
        return {'FINISHED'} if removed else {'CANCELLED'}


class KINARA_OT_StreamBake(bpy.types.Operator):
    bl_idname = 'kinara_bone_stream.bake'
    bl_label = 'Bake'
    bl_description = 'Bake all recorded motion to the selected streaming rig and detach it'

    @classmethod
    def poll(cls, context):
        return receiver.bake_target(context) is not None

    def execute(self, context):
        try:
            receiver.request_bake(context)
        except Exception as error:
            self.report({'WARNING'}, str(error))
            return {'CANCELLED'}
        return {'FINISHED'}


class KINARA_PT_StreamPanel(bpy.types.Panel):
    bl_label = 'Kinara Motion Stream'
    bl_idname = 'KINARA_PT_stream_panel'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Kinara'

    def draw(self, context):
        layout = self.layout
        assert layout is not None
        layout.prop(context.window_manager, 'kinara_bone_stream_port')
        layout.operator('kinara_bone_stream.disconnect' if receiver.sock else 'kinara_bone_stream.connect')
        layout.operator('kinara_bone_stream.bake')
        clean_row = layout.row()
        clean_row.operator_context = 'INVOKE_DEFAULT'
        clean_row.operator('kinara_bone_stream.clean')
        layout.label(text=receiver.status)
        if receiver.bake_status:
            layout.label(text=receiver.bake_status)


CLASSES = (
    KINARA_OT_StreamConnect,
    KINARA_OT_StreamDisconnect,
    KINARA_OT_StreamClean,
    KINARA_OT_StreamBake,
    KINARA_PT_StreamPanel,
)
