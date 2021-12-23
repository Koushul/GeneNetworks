import requests
import json 
import pandas as pd

class Enrichr(object):
    
    def __init__(self):
        self.ENRICHR_URL_ADDLIST = 'https://maayanlab.cloud/Enrichr/addList'
        self.ENRICHR_URL = 'https://maayanlab.cloud/Enrichr/enrich'
        self.QUERY_STR = '?userListId=%s&backgroundType=%s'
        self.libraries = [
            'VirusMINT', 
            'GO_Biological_Process_2021', 
            'MSigDB_Hallmark_2020', 
            'KEGG_2021_Human', 
            'Reactome_2016']
        
    
    def _addlist(self, module_number, geneset):
        genes_str = '\n'.join(geneset)
        description = ''
        payload = {
            'list': (None, genes_str),
            'description': (None, description)
        }
        response = requests.post(self.ENRICHR_URL_ADDLIST, files=payload)
        data = json.loads(response.text)
        
        return data['userListId']
    
    def get_enrichment_results(self, geneset, gene_set_library = 'GO_Biological_Process_2021'):
        user_list_id = self._addlist(geneset)
        response = requests.get(
        self.ENRICHR_URL + self.QUERY_STR % (user_list_id, gene_set_library))
        data = json.loads(response.text)
        df = pd.DataFrame(data[gene_set_library])[[1, 2, 3, 4, 5, 6]]
        df.columns = ['Terms', 'Pval', 'OddsRatio', 'Score', 'Genes', 'AdjPval']
        
        return df[round(df.AdjPval, 3) < 0.05].sort_values(by='AdjPval')