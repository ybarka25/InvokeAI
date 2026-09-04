"""Tags a control video with an application strength and stacking order for `vace_control_blend`."""

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import InputField, VideoField
from invokeai.app.invocations.primitives import VaceControlLayerOutput
from invokeai.app.services.shared.invocation_context import InvocationContext


@invocation(
    "vace_control_layer",
    title="Control Video Layer - VACE",
    tags=["video", "conditioning", "wan", "vace", "v2v", "compositing"],
    category="conditioning",
    version="1.0.0",
    classification=Classification.Prototype,
)
class VaceControlLayerInvocation(BaseInvocation):
    """Wraps a control video with an application strength and stacking order.

    Feed several of these into `vace_control_blend` to composite multiple control
    modalities (e.g. canny + depth + pose) into one control video for `wan_vace_video_encode`.
    `order` sets where this layer paints in the stack (lower paints first/further back);
    ties keep the layers' wiring order in the blend node's list. `strength` is this
    layer's opacity when painted over what's already there (1.0 fully replaces, 0.0 has
    no effect).
    """

    video: VideoField = InputField(description="The control video for this layer (e.g. output of a preprocessor node).")
    strength: float = InputField(
        default=1.0, ge=0.0, le=1.0, description="Application force (opacity) for this layer."
    )
    order: int = InputField(
        default=0, description="Stacking position (lower paints first/further back). Ties keep wiring order."
    )

    def invoke(self, context: InvocationContext) -> VaceControlLayerOutput:
        return VaceControlLayerOutput.build(video_name=self.video.video_name, strength=self.strength, order=self.order)
