from models.encoders.baseline_enc.utils import BASE_TIMM_VIT_MODEL, BaselineEncoder


class MVP(BaselineEncoder):
    def __init__(self, model_name="mvp-vitb-mae-egosoup", **kwargs):
        super().__init__(model_name=model_name, **kwargs)

    def _process_model_name(self, model_name):
        assert model_name.startswith("mvp-")
        model_basename = model_name.split("-")[1]
        return BASE_TIMM_VIT_MODEL[model_basename]

    def _process_state_dict(self, state_dict):
        return state_dict["model"]
