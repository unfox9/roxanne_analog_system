from motor.schema.action_schema import ActionSchema

def load_action_channels(path: str):
    return ActionSchema.from_mapping_file(path, strict_contiguous=False).channels
