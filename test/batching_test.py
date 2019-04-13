#!/usr/bin/env python3

import re

from absl.testing import absltest
from absl import logging

from ypack.batching import StatsAggregator

class UnitTests(absltest.TestCase):
    def test_stats_aggregator(self):
        stats = StatsAggregator()
        stats.add_batch('a', [1,2])
        stats.add_batch('a', [4,5])
        stats.add_batch('b', [7,8])

        self.assertCountEqual(stats.tags, ['a', 'b'])
        self.assertCountEqual([t for t in stats], ['a', 'b'])
        self.assertSequenceEqual(stats['a'].tolist(), [1, 2, 4, 5])

if __name__ == '__main__':
    absltest.main()
