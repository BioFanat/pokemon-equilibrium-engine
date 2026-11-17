from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'showdown'))
from ladder_metrics import summarize_games


class LadderMetricTests(unittest.TestCase):
    def test_elo_change_uses_completion_order_and_deduplicates_games(self):
        a=dict(battle_tag='a',result='win',rating=dict(before=1470,after=1495))
        b=dict(battle_tag='b',result='loss',rating=dict(before=1495,after=1479))
        r=summarize_games([a,b,b],2)
        self.assertEqual((r['elo_start'],r['elo_end'],r['elo_change']),(1470,1479,9))
        self.assertEqual(r['completed'],2); self.assertTrue(r['complete'])
        self.assertEqual(r['primary_metric'],'ladder_elo_change')
        self.assertEqual(r['results'],dict(win=1,loss=1))

    def test_pending_results_do_not_invent_an_elo(self):
        r=summarize_games([],50)
        self.assertIsNone(r['elo_change']); self.assertFalse(r['complete'])
        self.assertIsNone(r['win_rate'])


if __name__=='__main__': unittest.main()
