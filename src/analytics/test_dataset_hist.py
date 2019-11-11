from unittest import TestCase

from src.analytics.analytics import dataset_hist


class TestDataset_hist(TestCase):
    labels = ['q', 'e', 'q', 'r', 'r', 's', 'q']
    path = './'
    name = 'Validation'
    dataset_hist(labels, name, path)
