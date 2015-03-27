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

                                    
range R = 1..8; 
var{int} row[R] in R; 
solve { 
   forall(i in R,j in R: i < j) { 
      row[i] ≠  row[j]; 
      row[i] ≠  row[j] + (j - i); 
      row[i] ≠  row[j] - (j - i); 
   } 
}

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

Magic Series and Reification
int n = 5; 
range D = 0..n-1; 
var{int} series[D] in D; 
solve { 
   forall(k in D) 
     series[k] = sum(i in D) (series[i]=k); 
}

