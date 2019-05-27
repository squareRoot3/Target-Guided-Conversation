import os
data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'source')

class dts_ConvAI2(object):
    def __init__(self, path=data_path):
        self.path = path

    def _txt_to_json(self, txt_path, mode, cands):
        def pop_one_sample(lines):
            self_persona = []
            other_persona = []
            dialog = []
            candidates = []

            started = False
            while len(lines) > 0:
                line = lines.pop()
                id, context = line.split(' ', 1)
                id = int(id)
                context = context.strip()

                if started == False: # not started
                    assert id == 1
                    started = True
                elif id == 1: # break for next
                    lines.append(line)
                    break

                if context.startswith('partner\'s persona: '): # partner
                    assert mode in ['both', 'other']
                    other_persona.append(context[19:])

                elif context.startswith('your persona: '): # self
                    assert mode in ['both', 'self']
                    self_persona.append(context[13:])

                elif cands == False: # no cands 
                    try:
                        uttr, response = context.split('\t', 2)[:2]
                        dialog.append(uttr)
                        dialog.append(response)
                    except:
                        uttr = context 
                        dialog.append(uttr)
                else:
                    uttr, response, _, negs = context.split('\t', 4)[:4]
                    dialog.append(uttr)
                    dialog.append(response)                    
                    candidates.append(negs.split('|'))
                    candidates.append(None)

            return {
                'self_persona': self_persona,
                'other_persona': other_persona,
                'dialog': dialog,
                'candidates': candidates
            }

        lines = open(txt_path, 'r').readlines()[::-1]

        samples = []
        while len(lines) > 0:
            samples.append(pop_one_sample(lines))

        return samples

    def get_data(self, mode='train', revised=False, cands=False):
        txt_path = os.path.join(self.path, '{}_{}_{}{}.txt'.format(
            mode,
            'none',
            'revised' if revised is True else 'original',
            '' if cands is True else '_no_cands'))
        assert mode in ['train', 'valid', 'test', 'all']
        print("Get dialog from ", txt_path)
        assert os.path.exists(txt_path)
        return self._txt_to_json(txt_path, mode, cands)

    def get_dialogs(self, mode='all'):
        dialogs = [sample['dialog'] for sample in self.get_data(mode, False, False)]
        return dialogs