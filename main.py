import joblib
import networkx as nx
import json
import r2pipe, os, sys
import numpy as np
import joblib as jb
import csv
from utils import parameter_parser
from utils import write_output

inValid=0
def create_graph(path):
    r2 = r2pipe.open(path)
    r2.cmd('aaaa')
    data = r2.cmd('agCd')

    label={}
    G=nx.DiGraph()
    for lines in data.split('\n'):
        tmp=[]
        for words in lines.split():
            if words[0]=='"':
                words=words.replace('"','')
            tmp.append(words)    
        try:
            if tmp[1][1]=='l':
                func=tmp[1][7:]
                func=func.replace('"','')
                label[tmp[0]]=func
        except:
            pass
       
       
       
    for lines in data.split('\n'):
        tmp=[]
        for words in lines.split():
            if words[0]=='"':
                words=words.replace('"','')
            tmp.append(words)
        try:
            if tmp[1]=='->':
                G.add_edge(label[tmp[0]],label[tmp[2]])
        except:
            pass
    return G

def get_density(G):
    global inValid
    degree = {d[0]:d[1] for d in G.degree(G.nodes())}
    
    if len(degree)==0:
      inValid=1
    elif len(degree)==1:
      inValid=1
    else:
      inValid=0
    if inValid==0:
      density = (sum(degree.values())/(len(degree)-1)) / len(degree)
    else:
      density=-1


    return density

def shortest_path(G):

    # shortest_path=dict(nx.all_pairs_shortest_path(G.to_undirected()))
    # shortest_path_length={}

    # for start in shortest_path:
    #     tmp={}
    #     for target in shortest_path[start]:
    #         if start!=target:
    #             tmp[target]=len(shortest_path[start][target])
    #     shortest_path_length[start]=tmp

    List=[]
    for C in (G.subgraph(c).copy() for c in nx.connected_components(G.to_undirected())):
        List.append(nx.average_shortest_path_length(C))
    shortest_path=[]
    shortest_path.append(np.mean(List))
    shortest_path.append(np.max(List))
    shortest_path.append(np.min(List))
    shortest_path.append(np.median(List))
    shortest_path.append(np.std(List))

    return shortest_path

def closeness_centrality(G):

    List=list(nx.closeness_centrality(G).values())
    closeness_centrality=[]
    closeness_centrality.append(np.mean(List))
    closeness_centrality.append(np.max(List))
    closeness_centrality.append(np.min(List))
    closeness_centrality.append(np.median(List))
    closeness_centrality.append(np.std(List))
    
    return closeness_centrality

def betweeness_centrality(G):

    List=list(nx.betweenness_centrality(G.to_undirected()).values())
    betweeness_centrality=[]
    betweeness_centrality.append(np.mean(List))
    betweeness_centrality.append(np.max(List))
    betweeness_centrality.append(np.min(List))
    betweeness_centrality.append(np.median(List))
    betweeness_centrality.append(np.std(List))

    return betweeness_centrality

def degree_centrality(G):

    List=list(nx.degree_centrality(G).values())
    degree_centrality=[]
    degree_centrality.append(np.mean(List))
    degree_centrality.append(np.max(List))
    degree_centrality.append(np.min(List))
    degree_centrality.append(np.median(List))
    degree_centrality.append(np.std(List))

    return degree_centrality


def Feature_extraction(path):
    G=create_graph(path)
    global inValid
    feature=[]
    # append #nodes & # edges
    feature.append(G.number_of_nodes())
    feature.append(G.number_of_edges())

    # append Density
    feature.append(get_density(G))
    if inValid==0:
    # append Closeness Centrality
      for i in closeness_centrality(G):
          feature.append(i)

    # append Betweeness Centrality
      for i in betweeness_centrality(G):
          feature.append(i)

    # append Degree Centrality
      for i in degree_centrality(G):
          feature.append(i)

    # append Shortest Path
      for i in shortest_path(G):
          feature.append(i)

    return np.array(feature)

def main(args):
    global inValid
    label_dict = {'BenignWare', 'Malware'}
    if args.classify:
      label_dict = {'Mirai', 'Tsunami', 'Hajime', 'Dofloo', 'Bashlite', 'Xorddos', 'Android', 'Unknown'}
    feature=Feature_extraction(args.input_path)
    if inValid==0:
      if args.classify:
        Model = jb.load('./Model_classification/GraphTheory_' + args.model) 
      else:
        Model = jb.load('./Model_detection/GraphTheory_detection_' + args.model)
      result = Model.predict_proba(feature.reshape(1, -1))
      print(result)
      write_output(args.input_path, args.output_path, result[0], label_dict)
    else:
      result = [-1]
      write_output(args.input_path, args.output_path, result, label_dict)
      print("Can't extract feature.")    
    
if __name__=='__main__':
    args=parameter_parser()
    main(args)