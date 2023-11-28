from models.encoders.baseline_enc.utils import BaselineEncoder


def remove_language_head(state_dict):
    keys = state_dict.keys()
    new_state_dict = {}
    ## Hardcodes to remove the language head
    for key in list(keys):
        if ("lang_enc" in key) or ("lang_rew" in key):
            del state_dict[key]
        else:
            new_state_dict[key[len("module.convnet.") :]] = state_dict[key]
    return new_state_dict


class R3M(BaselineEncoder):
    def __init__(self, model_name="r3m-resnet34", **kwargs):
        super().__init__(model_name=model_name, **kwargs)

    def _process_model_name(self, model_name):
        assert model_name.startswith("r3m-")
        return model_name[len("r3m-") :]

    def _process_state_dict(self, state_dict):
        return remove_language_head(state_dict["r3m"])
