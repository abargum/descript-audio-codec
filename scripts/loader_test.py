from rave.rave_model import RAVE
import cached_conv as cc

if __name__ == "__main__":

    cc.use_cached_conv(True)

    generator = RAVE()

    kwargs = {
            "folder": f"runs/baseline/latest/",
            "map_location": "cpu",
            "package": False,
        }

    generator, g_extra = generator.load_from_folder(**kwargs)

    print(generator)