# Case 1
################################################################################
from bs4 import BeautifulSoup
import requests
import pandas as pd
url='https://gdc.cancer.gov/resources-tcga-users/tcga-code-tables/tissue-source-site-codes'
# One method using BeautifulSoup
request_html = requests.get(url)
soup = BeautifulSoup(request_html.content, 'html.parser')
Tables=soup.findAll("table")[1]
output_rows = []
for table_row in Tables.findAll('tr'):
    columns = table_row.findAll(['td','th'])
    output_row = []
    for column in columns:
        output_row.append(column.text)
    output_rows.append(output_row)
Table_df=pd.DataFrame(output_rows)
Table_df.columns = Table_df.iloc[0]
Table_df.to_csv("Tissue_Source_Site_Codes.txt",index=FALSE)


# Case 2
################################################################################
import pandas as pd
url='https://gdc.cancer.gov/resources-tcga-users/tcga-code-tables/tissue-source-site-codes'
df_list = pd.read_html(url)
Tables_pd=df_list[1]
Tables_pd.to_excel('Tissue_Source_Site_Codes_from_pandas.xls',index='false')
################################################################################


#Case 3
################################################################################
from bs4 import BeautifulSoup
import requests
html_cbioportal = 'https://www.cbioportal.org'
soup_cbioportal = BeautifulSoup(requests.get(html_cbioportal).content, 'html.parser')
print("Title of the document:")
print(soup_cbioportal.find("title").get_text())
################################################################################


# Case 4
################################################################################
# urllib is a package that collects several modules for working with URLs:
import urllib
from bs4 import BeautifulSoup
import pandas as pd
import re
###############################################
def get_uniprot (query='',query_type='PDB_ID'):
    #query_type must be: "PDB_ID" or "ACC"
    url = 'https://www.uniprot.org/uploadlists/' #This is the webser to retrieve the Uniprot data
    params = {
    'from':query_type,
    'to':'ACC',
    'format':'txt',
    'query':query
    }

    data = urllib.parse.urlencode(params)
    data = data.encode('ascii')
    request = urllib.request.Request(url, data)
    with urllib.request.urlopen(request) as response:
        res = response.read()
        page=BeautifulSoup(res).get_text()
        page=page.splitlines()
    return page
###############################################
prots=['P53_HUMAN','P53_MOUSE','P53_RAT']
table=pd.DataFrame()
for index,entry in enumerate(prots):
    sizes=[]
    pdbs=[]
    funtions=[]
    process=[]
    organism=[]
    data=get_uniprot(query=entry,query_type='ACC')

    table.loc[index,'Uniprot_entry']=entry

    for line in data:
        if 'OS   ' in line:
            line=line.strip().replace('OS   ','').replace('.','')
            organism.append(line)
            table.loc[index,'Organism']=(", ".join(list(set(organism))))
        if 'ID   ' in line:
            line=re.sub('ID\W+Reviewed;\W+','',line.strip())
            sizes.append(line)
            table.loc[index,'Sizes']=(", ".join(list(set(sizes))))

        if 'DR   PDB;' in line:
            line=line.strip().replace('DR   ','').replace(';','')
            pdbs.append ((line.split()[1]+':'+line.split()[3]))
            table.loc[index,'PDB:Resol']=(", ".join(list(set(pdbs))))

        if 'DR   GO; GO:' in line:
            line=line.strip().replace('DR   GO; GO:','').replace(';','').split(':')
            if 'F' in line[0]:
                funtions.append(line[1])
                table.loc[index,'GO_funtion']=(", ".join(list(set(funtions))))
            else:
                process.append (line[1])
                table.loc[index,'GO_process']=(", ".join(list(set(process))))
################################################################################


# Case 5
################################################################################
import requests
import sys
import json
requestURL = "https://www.ebi.ac.uk/proteins/api/proteins?offset=0&size=100&accession=Q9Y4C1"
r = requests.get(requestURL, headers={ "Accept" : "application/json"})
if not r.ok:
    r.raise_for_status()
    sys.exit()
json_data=json.loads(r.text)
print(json_data[0])
################################################################################
