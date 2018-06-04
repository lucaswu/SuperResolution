# data path and log path
ORIGINAL_IMAGES_PATH = '../data/originals'
TRAINING_DATA_PATH = '../data/train'
VALIDATION_DATA_PATH = '../data/valid'
TESTING_DATA_PATH = '../data/test'
INFERENCES_SAVE_PATH = '../inferences'
TRAINING_SUMMARY_PATH = '../training_summary'
CHECKPOINTS_PATH = '../checkpoints'
MAX_CKPT_TO_KEEP = 50       # max checkpoint files to keep

# patch generation
PATCH_SIZE = 80             # must be even, the image size croped from original images
PATCH_GEN_STRIDE = 32       # maybe used by data generation
PATCH_RAN_GEN_RATIO = 2     # the number of random generated patches is max(img.height, img.width) // PATCH_RAN_GEN_RATIO

# model and training
MODEL_NAME = 'vgg_deconv_7'             # srcnn, vgg7, vgg_deconv_7
BATCH_SIZE = 16
INPUT_SIZE = 28                         # the image size input to the network
SCALE_FACTOR = 2
LABEL_SIZE = SCALE_FACTOR * INPUT_SIZE  # the high resolution image size used as label
NUM_CHENNELS = 3

# data queue
MIN_QUEUE_EXAMPLES = 1024
NUM_PROCESS_THREADS = 3
NUM_TRAINING_STEPS = 1000000
NUM_TESTING_STEPS = 600

# data argumentation
MAX_RANDOM_BRIGHTNESS = 0.2
RANDOM_CONTRAST_RANGE = [0.8, 1.2]
GAUSSIAN_NOISE_STD = 0.01  # [0...1] (float)
JPEG_NOISE_LEVEL = 2    # [0...4] (int)
