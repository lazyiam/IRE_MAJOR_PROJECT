from __future__ import print_function
from collections import *
import requests
import json
tailset = set()
head_entities = defaultdict(set)

def getscore(hq):
    headset = head_entities[hq]
    common_entites=tailset.intersection(headset)

    entity_score=0
    if (len(headset)+len(tailset)-len(common_entites))==0:
        entity_score =0


    else:
        entity_score= float(len(common_entites))/(len(headset)+len(tailset)-len(common_entites))
    return entity_score

allheads = []
fname = "testhead.txt"

with open(fname) as FF:
    url = "http://localhost:8080/dexter-webapp/api/rest/annotate"
    otherParams = "n=50&wn=false&debug=false&format=text&min-conf=0.5"
    cnt =0
    for line in FF:
        cnt +=1
        hq = line.strip()
        textParam = hq
        finalURL = url + "?text=" + textParam + "&" + otherParams
        response = requests.get(finalURL)
        pairs = open("ggh.json",'w')
        print(response.content,file=pairs)
        pairs.close()
        jsonFile = open('ggh.json', 'r')
        values = json.load(jsonFile)
        jsonFile.close()
        if hq not in head_entities:
            head_entities[hq]  = set()

        for i in range(len(values['spots'])):
            head_entities[hq].add(values['spots'][i]['entity'])
        allheads.append(line.strip())
        print(cnt)
    print ("All headds done")

fname = "tail5.txt"
myfl = open("enity_linked_pairs.txt",'w')
with open(fname) as FF:
    for tq in FF:
        tailset.clear()
        tq = tq.strip()
        url = "http://localhost:8080/dexter-webapp/api/rest/annotate"
        otherParams = "n=50&wn=false&debug=false&format=text&min-conf=0.5"
        textParam = tq
        finalURL = url + "?text=" + textParam + "&" + otherParams
        response = requests.get(finalURL)
        pairs = open("ggt.json",'w')
        print(response.content,file=pairs)
        pairs.close()
        jsonFile = open('ggt.json', 'r')
        values = json.load(jsonFile)
        jsonFile.close()
        tailset = set()
        for i in range(len(values['spots'])):
            tailset.add(values['spots'][i]['entity'])
        maxscr = 0.0
        maxhead =''
        for i in range(len(allheads)):
            # print (allheads[i]," || ",tq.strip())

            x = getscore(allheads[i])
            # print (x)
            if x > maxscr and x > 0.0:
                maxscr = x
                maxhead =  allheads[i]
        if maxhead!='':
            toprint = ''
            toprint = str(tq) + " || "+ str(maxhead)
            print ("done")
            print (toprint,file = myfl)
