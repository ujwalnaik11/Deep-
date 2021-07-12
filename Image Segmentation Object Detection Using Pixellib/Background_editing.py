import pixellib
from pixellib.tune_bg import alter_bg

change_bg = alter_bg(model_type = "pb")
change_bg.load_pascalvoc_model("xception_pascalvoc.pb")
change_bg.blur_bg("cycle.jpg", extreme = True, detect = "person", output_image_name="blur_img.jpg")
