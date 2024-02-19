"""fdf dataset."""

import fdf_dataset_builder
import tensorflow_datasets as tfds

class FdfTest(tfds.testing.DatasetBuilderTestCase):
  """Tests for dataset."""
  DATASET_CLASS = fdf_dataset_builder.Builder
  SPLITS = {
      'train': 8,  # Number of fake train example
  }

  # If you are calling `download/download_and_extract` with a dict, like:
  #   dl_manager.download({'some_key': 'http://a.org/out.txt', ...})
  # then the tests needs to provide the fake output paths relative to the
  # fake data directory
  # DL_EXTRACT_RESULT = {'some_key': 'output_file1.txt', ...}
  DL_EXTRACT_RESULT = {
    'metadata': ''
  }

if __name__ == '__main__':
  tfds.testing.test_main()
