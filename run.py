import click
import mpc
import datetime
import json


@click.command()
@click.option('--name', prompt='Dataset name(str)', help='The name of the loaded dataset')
@click.option('--theta', prompt='theta(int)', help='The value of the parameter theta')
@click.option('--k', prompt='k(int)', help='The value of the parameter k')
@click.option('--method', prompt='Type one number to chose the algorithm: [1]MPCKC; [2]MPCWC; [3]MPCSC. (int)', help='Three periodic cliques mining algorithms')
def doit(name, theta, k, method):
    theta = int(theta)
    k = int(k)
    G = mpc.Graph(name)
    G.prepare(theta, k)
    starttime = datetime.datetime.now()
    if method == "1":
        result = G.MPCKC(theta, k)
    if method == "2":
        result = G.MPCWC(theta, k)
    if method == "3":
        result = G.MPCSC(theta, k)
    endtime = datetime.datetime.now()
    interval = (endtime - starttime)
    print("All time:" + str(interval))
    print('New nodes stored in file "NEWNODES.json" : {new_node_id: [raw_node_id, [starttime, interval]]}')
    print("if theta=4, one node [1, [2003, 1]] means node id 1 is periodic at 2003,2004,2005,2006")
    print("if theta=3, one node [2, [2003, 2]] means node id 2 is periodic at 2003,2005,2007")
    newnode ={}
    for _key, _value in result[0].items():
        newnode[_value] = _key
    file_name = "NEWNODES.json"
    file_object = open(file_name, 'w')
    file_object.write(json.dumps(newnode))
    file_object.close()
    print("MPCliques:")
    print(G.maximalclique)


if __name__ == '__main__':
    doit()
