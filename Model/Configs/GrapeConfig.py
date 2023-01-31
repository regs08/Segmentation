# define a configuration for the model
####
#calculating steps per epoch as seen here:
#https://stackoverflow.com/questions/63437552/steps-per-epoch-validation-steps-in-matterport-mask-rcnn
####

from mrcnn.config import Config

class GrapeConfig(Config):
    # define the name of the configuration
    NAME = "grape_cfg_coco"
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1
    GPU_COUNT = 1
    # number of training steps per epoch
    DETECTION_MIN_CONFIDENCE = 0.9 # Skip detections with < 90% confidence

    def get_attr_as_dict(self, save=False, out_file=""):
        """
        using the same method as display we return a dict instead in order to save the params
        :return:
        """
        a_dict = {}
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                a_dict[a] = getattr(self, a)
        if save:
            assert out_file, print(f'Invalid out file')
            import pickle
            with open(out_file, 'wb') as f:
                pickle.dump(a_dict, f)

        return a_dict
