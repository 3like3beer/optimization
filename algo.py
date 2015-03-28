#Dp naive
int O(int k,int j) { 
   if (j == 0) 
     return 0; 
   else if (wj <= k) 
     return max(O(k,j-1),vj + O(k-wj,j-1)); 
   else 
     return O(k,j-1) 
}

#DP real
– start with zero items?
– continue with one item?
– then two items?
– ...?
– then all items


#CP coloring
enum Countries = { Belgium, Denmark, France, Germany,  
                   Netherlands, Luxembourg }; 
enum Colors = { black, yellow, red, blue }; 
var{Colors} color[Countries]; 
?
solve { 
color[Belgium] ≠ color[France]; 
color[Belgium] ≠ color[Germany]; 
color[Belgium] ≠ color[Netherlands]; 
color[Belgium] ≠ color[Luxembourg]; 
color[Denmark] ≠ color[Germany]; 
color[France] ≠ color[Germany]; 
color[France] ≠ color[Luxembourg]; 
color[Germany] ≠ color[Netherlands]; 
color[Germany] ≠ color[Luxembourg]; 
}

#CP propogation engine
propagate() 
{ 
  repeat 
    select a constraint c; 
    if c is infeasible given the domain store then 
      return failure; 
    else 
      apply the pruning algorithm associated with c; 
  until no constraint can remove any value from the  
  domain of its variables; 
  return success; 
}


##Send More Money
enum Letters = { S, E, N, D, M, O, R, Y};
range Digits = 0..9;
var{int} value[Letters] in Digits;
var{int} carry[1..4] in 0..1;
solve {
forall(i in Letters, j in Letters: i < j)
   value[i] ≠ value[j];
value[S] ≠ 0;
value[M] ≠ 0;
carry[4]                       = value[M];
carry[3] + value[S] + value[M] = value[O] + 10 * carry[4];
carry[2] + value[E] + value[O] = value[N] + 10 * carry[3];
carry[1] + value[N] + value[R] = value[E] + 10 * carry[2];
           value[D] + value[E] = value[Y] + 10 * carry[1];
}
###Search space
forall(i in Letters, j in Letters: i < j)
   value[i] ≠ value[j];
value[S] ≠ 0;
value[M] ≠ 0;
carry[4]                       = value[M];


##Queen Problem
range R = 1..8; 
var{int} row[R] in R; 
solve { 
   forall(i in R,j in R: i < j) { 
      row[i] ≠  row[j]; 
      row[i] ≠  row[j] + (j - i); 
      row[i] ≠  row[j] - (j - i); 
   } 
}
#alldifferent
range R = 1..8;
var{int} row[R] in R;
solve {
   alldifferent(row);
   alldifferent(all(i in R) row[i]+i);
   alldifferent(all(i in R) row[i]-i); 
}

#The 8-Queens Problem with Dual Modeling
range R = 1..8; 
range C = 1..8; 
var{int} row[C] in R; 
var{int} col[R] in C 
solve { 
   forall(i in R,j in R: i < j) { 
      row[i] ≠  row[j]; 
      row[i] ≠  row[j] + (j - i); 
      row[i] ≠  row[j] - (j - i); 
   } 
   forall(i in C,j in C: i < j) { 
      col[i] ≠  col[j]; 
      col[i] ≠  col[j] + (j - i); 
      col[i] ≠  col[j] - (j - i); 
   } 
   forall(r in R,c in C) 
      (row[c] = r) <=> (col[r] = c); 
}

#A search procedure for the 8-queens problem
range R = 1..8;
var{int} row[R] in R;
solve {
   forall(i in R,j in R: i < j) {
      row[i] ≠  row[j];
      row[i] ≠  row[j] + (j - i);
      row[i] ≠  row[j] - (j - i);
   }
}
using {
   forall(r in R)
      tryall(v in R)
         row[r] = v;
}

range R = 1..8;
var{int} row[R] in R;
solve {
   forall(i in R,j in R: i < j) {
      row[i] ≠  row[j];
      row[i] ≠  row[j] + (j - i);
      row[i] ≠  row[j] - (j - i);
   }
}
using {
   forall(r in R) by row[r].getSize()
      tryall(v in R)
         row[r] = v;
}
#select first the variable with the smallest domain

#Dynamic orderings for variable and value choices
range R = 1..8;
range C = 1..8;
var{int} row[C] in R;
var{int} col[R] in C
solve {
   forall(i in R,j in R: i < j) {
      row[i] ≠  row[j];
      row[i] ≠  row[j] + (j - i);
      row[i] ≠  row[j] - (j - i);
   }
   forall(i in C,j in C: i < j) {
      col[i] ≠  col[j];
      col[i] ≠  col[j] + (j - i);
      col[i] ≠  col[j] - (j - i);
   }
   forall(r in R,c in C)
      (row[c] = r) <=> (col[r] = c);
}
using {
   forall(r in R) by row[r].getSize()
      tryall(v in R) by col[v].getSize()
         row[r] = v;
}

#CP model for The ESDD Deployment Problem
minimize 
    sum(a in C,b in C: a != b) f[a,b]*h[x[a],x[b]]
subject to {
   forall(S in Col,c1 in S,c2 in S: c1 < c2)        
     x[c1] = x[c2];    
   forall(S in Sep)       
     alldifferent(all(c in S) x[c]);
} 
using {  
 while (!bound(x))     
  selectMax(i in C:!x[i].bound(),j in C)(f[i,j])  
      tryall(n in N) by (min(l in x[j].memberOf(l)) h[n,l]) 
       x[i] = n;
}

#Magic Series and Reification
int n = 5; 
range D = 0..n-1; 
var{int} series[D] in D; 
solve { 
   forall(k in D) 
     series[k] = sum(i in D) (series[i]=k); 
}

int n = 5; 
range D = 0..n-1; 
var{int} series[D] in D; 
solve { 
   forall(k in D) { 
      var{int} b[D] in 0..1; 
      forall(i in D) 
         booleq(b[i],series[i],k); 
     series[k] = sum(i in D) b[i]; 
   } 
}
booleq(b,x,v) <=> (b=1 ^ x=v) v (b=0 ^ x6=v)

#Redundant Constraints (reduce Search Space)
int n = 5;
range D = 0..n-1;
var{int} series[D] in D;
solve {
   forall(k in D)
     series[k] = sum(i in D) (series[i]=k);
   sum(i in D) series[i] = n; 
   sum(i in D) i * series[i] = n;
}
#Market Split Problems
range C = ...;
range V = ...;
int w[C,V] = ...;
int rhs[C];
var{int} x[V] in 0..1;
solve {
   forall(c in C)
      sum(v in V) w[c,v] x[v] = rhs[c];
   sum(v in V) (sum(c in C) alphac * w[c,v]) * x[v] = sum(c in C) alphac * rhs[c];
}   


#Stable marriages
enum Men = {George,Hugh,Will,Clive}; 
enum Women = {Julia,Halle,Angelina,Keira}; 
int wrank[Men,Women]; 
int mrank[Women,Men]; 
... 
var{Women} wife[Men]; 
var{Men}   husband[Women];  
solve { 
   forall(m in Men) 
      husband[wife[m]] = m; 
   forall(w in Women) 
      wife[husband[w]] = w; 
   forall(m in Men, w in Women) 
      wrank[m,w] < wrank[w,wife[m]] => mrank[w,husband[w]] < mrank[w,m]; 
   forall(w in Women, m in Men) 
      mrank[w,m] < mrank[w,husband[m]] => wrank[m,wife[m]] < mrank[m,w]; 
}

#Sudoku
range R = 1..9;
var{int} s[R,R] in R;
solve {
 //constraints on fixed positions
 forall(i in R)
   alldifferent(all(j in R) s[i,j]);
 forall(j in R)
   alldifferent(all(i in R) s[i,j]);
 forall(i in 0..2,j in 0..2)
   alldifferent(all(r in i*3+1..i*3+3,
                    c in j*3+1..j*3+3) s[r,c]);
}

#BIBDs
range Rows = 1..v; 
range Cols = 1..b; 
var{int} m[Rows,Cols] in 0..1; 
solve { 
   forall(i in Rows) 
      sum(y in Cols) m[i,y] = r; 
   forall(j in Cols) 
      sum(x in Rows) m[x,j] = k; 
   forall(i in Rows,j in Rows: j > i) 
      sum(x in Cols) (m[i,x] & m[j,x]) = 1; 
}

#Sym breaking
range Rows = 1..v; 
range Cols = 1..b; 
var{int} m[Rows,Cols] in 0..1; 
solve { 
   forall(i in Rows) 
      sum(y in Cols) m[i,y] = r; 
   forall(j in Cols) 
      sum(x in Rows) m[x,j] = k; 
   forall(i in Rows,j in Rows: j > i) 
      sum(x in Cols) (m[i,x] & m[j,x]) = 1; 
   forall(i in 1..v-1) 
      lexleq(all(j in Cols) m[i,j],all(j in Cols) m[i+1,j]); 
   forall(j in 1..b-1) 
      lexleq(all(i in Rows) m[i,j],all(i in Rows) m[i,j+1]); 
}

#Scene Allocation
range Scenes = …; 
range Days  = …; 
range Actor = …; 
int fee[Actor] = …;  
set{Actor} appears[Scenes] = …; 
set{int} which[a in Actor] = setof(i in Scenes) member(a,appears[i]); 
var{int} shoot[Scenes] in Days; 
?
minimize 
   sum(a in Actor) sum(d in Days)  
      fee[a] * or(s in which[a]) (shoot[s]=d)  
subject to  
   atmost(all(i in Days) 5,Days,shoot); 


#Sym breaking
range Scenes = 1..n; 
range Days  = 1..m; 
range Actor = …; 
int fee[Actor] = …;  
set{Actor} appears[Scenes] = …; 
set{int} which[a in Actor] = setof(i in Scenes) member(a,appears[i]); 
var{int} shoot[Scenes] in Days; 
?
minimize 
   sum(a in Actor) sum(d in Days)  
      fee[a] * or(s in which[a]) (shoot[s]=d)  
subject to { 
   atmost(all(i in Days) 5,Days,shoot); 
   scene[1] = 1; 
   forall(s in Scenes: s > 1) 
      scene[s] <= max(k in 1..s-1) scene[k] + 1; 
}

#Car sequencing
range Slots = ...; 
range Configs = ...; 
range Options = ...; 
int demand[Configs] = ...; 
int nbCars = sum(c in Configs) demand[c]; 
int lb[Options] = ...; 
int ub[Options] = ...; 
int requires[Options,Config] = ...; 
var{int} line[Slots] in Configs; 
var{int} setup[Options,Slots] in 0..1; 
?
solve { 
   forall(c in Configs) 
      sum(s in Slots) (line[s] = c) = demand[c]; 
    
   forall(s in Slots,o in Options) 
      setup[o,s] = requires[o,line[s]]; 
?
   forall(o in Options, s in 1..nbCars-ub[o]+1) 
      sum(j in s..s+ub[o]-1) setup[o,s] <= lb[o]; 
}   

#Car Sequencing: Redundant Constraints
range Slots = ...; 
range Configs = ...; 
range Options = ...; 
int demand[Configs] = ...; 
int lb[Options] = ...; 
int ub[Options] = ...; 
int requires[Options,Config] = ...; 
var{int} line[Cars] in Configs; 
var{int} setup[Options,Slots] in 0..1; 
?
solve { 
   forall(c in Configs) 
      sum(s in Slots) (line[s] = c) = demand[c]; 
   forall(s in Slots,o in Options) 
      setup[o,s] = requires[o,line[s]]; 
   forall(o in Options, s in 1..nbCars-ub[o]+1) 
      sum(j in s..s+ub[o]-1) setup[o,s] <= lb[o]; 
    
   forall(o in Options, i in 1..demand[o]) 
      sum(s in 1..nbCars-i*ub[o]) setup[o,s] >= demand[o] - i*lb[o]; 
    
}   

#The perfect square problem
range R = 1..8; 
int s = 122; range Side = 1..s; range Square = 1..21 
int side[Square] = [50,42,37,35,33,29,27,25,24,19,18,17,16,15,11,9,8,7,6,4,2]; 
var{int} x[Square] in Side; 
var{int} y[Square] in Side; 
?
solveall { 
  forall(i in Square) { 
 x[i]<=s-side[i]+1;  
  y[i]<=s-side[i]+1;  
} 
  forall(i in Square,j in Square: i<j)    
    x[i]+side[i]<= x[j] || x[j]+side[j]<=x[i] ||  y[i]+side[i]<=y[j] || y[j]+side[j]<=y[i]
   
  forall(p in Side) { 
     sum(i in Square) side[i]*((x[i]<=p) && (x[i]>=p-side[i]+1)) = s;    
     sum(i in Square) side[i]*((y[i]<=p) && (y[i]>=p-side[i]+1)) = s;    
  } 
} 
redundant 
constraints
non-overlapping
constraints


using { 
   
  forall(p in Side)  
    forall(i in Square)  
      try  
         x[i] = p; 
      |   
     x[i] != p; 
?
  forall(p in Side)  
    forall(i in Square)  
      try 
         y[i] = p; 
      |  
         y[i] != p; 
?
}
choose a  x-coordinate p
consider a square i
decide whether to place 
i at position p

