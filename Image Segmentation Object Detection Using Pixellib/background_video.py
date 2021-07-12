import pixellib
from pixellib.tune_bg import alter_bg

change_bg = alter_bg(model_type = "pb")
change_bg.load_pascalvoc_model("xception_pascalvoc.pb")
change_bg.color_video("sample_video.mp4", colors =  (0, 128, 0), frames_per_second=10, output_video_name="output_video.mp4", detect = "person")
