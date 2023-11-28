from models.encoders.baseline_enc.utils import BASE_TIMM_VIT_MODEL, BaselineEncoder


class VC1(BaselineEncoder):
    def __init__(self, model_name="vc1-vitb", **kwargs):
        super().__init__(model_name=model_name, **kwargs)

    def _process_model_name(self, model_name):
        assert model_name.startswith("vc1-")
        return BASE_TIMM_VIT_MODEL[model_name[len("vc1-") :]]

    def _process_state_dict(self, state_dict):
        del state_dict["model"]["mask_token"]
        return state_dict["model"]
