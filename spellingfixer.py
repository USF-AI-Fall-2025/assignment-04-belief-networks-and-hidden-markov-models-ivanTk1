import math
from collections import defaultdict, Counter
#

def load_data(file):
    pairs = []
    with open(file) as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            correct, typos = line.split(":", 1)
            correct = correct.strip().lower()
            for typo in typos.strip().split():
                pairs.append((correct, typo.lower()))
    return pairs

#compute emission probabilities
def emit(p):
    c=defaultdict(Counter)
    #go through each apir and compare letters at same position(count emissions)
    for a,b in p:
        for i,j in zip(a,b):c[i][j]+=1
    e={}
    for i in c:
        t=sum(c[i].values())
        #normalize counts to probabilities
        e[i]={j:c[i][j]/t for j in c[i]}
    return e

#compute transition probabilities
def trans(p):
    c=defaultdict(Counter)
    #go through each pair and count transitions between letters
    for a,_ in p:
        r="^"
        for x in a:c[r][x]+=1;r=x
        c[r]["$"]+=1
    t={}
    for r in c:
        s=sum(c[r].values())
        #normalize counts to probabilities
        t[r]={x:c[r][x]/s for x in c[r]}
    return t

#Viterbi algorithm finds the most probable sequence of spelling corrections
def viterbi(w,e,t):
    L=list(e.keys())
    V=[{}];path={}
    for s in L:
        #calculate initial probabilities start transition and emission
        V[0][s]=math.log(e[s].get(w[0],1e-6))+math.log(t["^"].get(s,1e-6))
        path[s]=[s]
    #iterate through the rest of the observed letters
    for i in range(1,len(w)):
        V.append({});np={}
        #for each possible current state
        for s in L:
            #emission probability
            em=math.log(e[s].get(w[i],1e-6))
            #previous best state
            best=max((V[i-1][p]+math.log(t[p].get(s,1e-6))+em,p)for p in L)
            V[i][s]=best[0];np[s]=path[best[1]]+[s]
        path=np
        #find the best last state and backtrack to get the full path
    last=max((V[-1][s]+math.log(t[s].get("$",1e-6)),s)for s in L)
    return "".join(path[last[1]])

def main():
    d=load_data("aspell.txt")
    e=emit(d);t=trans(d)
    while True:
        x=input("\nEnter text (or 'quit'): ").strip()
        if x.lower()=="quit":break
        print("Corrected:", " ".join(viterbi(w,e,t)for w in x.split()))

if __name__=="__main__":main()
