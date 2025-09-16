import pandas as pd
import csv
import spacy
import swifter
import os


spacy.prefer_gpu()
if __name__ == '__main__':
    finnish = pd.read_csv(os.path.expanduser('~/defgen_formatted/test_axolotl24st_fi.tsv'), sep='\t')
    print(finnish['example'].apply(len).max())
    nlp = spacy.load("de_core_news_sm")


    def select_sent(x):
        if len(x['example']) > 303:
            sents = nlp(x['example'])
            start, end = map(int, x['indices_target_token'].split(':'))
            useful = []
            for sent in sents.sents:
                if start >= sent.start_char and end <= sent.end_char:
                    useful.append(sent.text)


            useful = ' '.join(useful)
            if not useful:
                return x['word']
        else:
            return x['example']
        return useful
    data = pd.read_csv('../../shared_task_LChange24/data/axolotl.test.surprise.gold.tsv', sep='\t', quoting=csv.QUOTE_NONE)
    new_data = data.swifter.apply(select_sent, axis=1)
    data = data[['word']]
    data['example'] = new_data
    prompt = ". Was ist die Definition von <TRG>?"
    data['example_with_prompt'] = data.swifter.apply(lambda x: x['example'].rstrip(
    ).rstrip('.').rstrip() + prompt.replace(
        '<TRG>', x['word']), axis=1)
    data = data.rename(columns={'word': 'target'})
    pd.DataFrame(data).to_csv(os.path.expanduser('~/defgen_formatted/test_axolotl24st_de.tsv.gz'),
                              sep='\t', compression='gzip', index=False)