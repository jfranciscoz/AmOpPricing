# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 19:53:23 2020

The equations come from the paper "Efficient Analytic Approximation of American
Option Values" from Barone-adesi, 1987.

@author: Francisco Zambrano (2653108)
"""
###########################################################
### Imports
import numpy as np
import pandas as pd
import scipy.stats as st

###########################################################
### vq= roots(db, dr, dT, ds)
def roots(db, dr, dT, ds):
    '''
    roots of equation (13)
    '''
    
    dN= 2*db/ds**2
    dM= 2*dr/ds**2
    dK= 1 - np.exp(-dr*dT)
    
    a= 1
    b= dN - 1
    c= -dM/dK
    
    [q_2, q_1]= np.roots([a, b, c])
    
    return [q_1, q_2]

###########################################################
### [dCall_seed, dPut_seed]= SeedPrices(lq_inf, db, dT, ds, iX)
def SeedPrices(lq_inf, db, dT, ds, iX):
    ''' 
    Initiation value for finding the critical comodity price throughout the
    iterative procedure (Section II, A).
    '''
    [q_1, q_2]= lq_inf
    #eq (30)
    dCall_S_inf= iX/(1-1/q_2)
    #eq (32)
    dPut_S_inf=  iX/(1-1/q_1)
    
    dh_1= (db*dT - 2*ds*np.sqrt(dT)) * (iX/(iX-dPut_S_inf))
    dh_2= -(db*dT + 2*ds*np.sqrt(dT)) * (iX/(dCall_S_inf-iX))
    
    #eq (31)
    dCall_seed= iX + (dCall_S_inf - iX)*(1 - np.exp(dh_2))
    #eq (33)
    dPut_seed=  dPut_S_inf + (iX - dPut_S_inf)*np.exp(dh_1)
    
    return [dCall_seed, dPut_seed]

###########################################################
### dCrit_Call= Critical_Call_Price(lSeedPrices, lq, iX, db, ds, dT, dr)
def Critical_Call_Price(lSeedPrices, lq, iX, db, ds, dT, dr):
    ''' 
    Finding the critical commodity price throughout the iterative procedure.
    '''
    [dCrit_Call, dCrit_Put]= lSeedPrices
    [dq_1, dq_2]=            lq
    
    dError= 1
    while dError > 0.00001:
        # print('Critical Price :',dCrit_Call)
        d_1= (np.log(dCrit_Call/iX) + (db + 0.5*ds**2)*dT)/(ds*np.sqrt(dT))
        d_2= d_1 - ds*np.sqrt(dT)
        
        #eq (5)
        dEU_Call= dCrit_Call*np.exp((db-dr)*dT)*st.norm.cdf(d_1)\
            - iX*np.exp(-dr*dT)*st.norm.cdf(d_2)
        
        #Condition to satisfy eq (19)
        #eq (26b)
        dRHS= dEU_Call + (1-np.exp((db-dr)*dT)*st.norm.cdf(d_1))*dCrit_Call/dq_2
        #eq (26a)
        dLHS= dCrit_Call - iX
        
        #Condition value
        dError= abs(dLHS-dRHS)/iX
        # print('Relative Abs. Error: ',dError)
        
        #eq (27) Finding next guess price:
        db_i= np.exp((db-dr)*dT)*st.norm.cdf(d_1)*(1 - 1/dq_2)\
            + (1 - np.exp((db-dr)*dT)*st.norm.pdf(d_1)/ds*np.sqrt(dT))/dq_2
        
        #eq (28)
        dCrit_Call= (iX + dRHS - db_i*dCrit_Call)/(1 - db_i)
        
    return dCrit_Call

###########################################################
### dCrit_Put= Critical_Put_Price(lSeedPrices, lq, iX, db, ds, dT, dr)
def Critical_Put_Price(lSeedPrices, lq, iX, db, ds, dT, dr):
    ''' 
    Finding the critical commodity price throughout the iterative procedure.
    '''
    [dCrit_Call, dCrit_Put]= lSeedPrices
    [dq_1, dq_2]=            lq
    
    dError= 1
    while dError > 0.00001:
        d_1= (np.log(dCrit_Put/iX) + (db + 0.5*ds**2)*dT)/(ds*np.sqrt(dT))
        d_2= d_1 - ds*np.sqrt(dT)
        
        #eq (6)
        dEU_Put= iX*np.exp(-dr*dT)*st.norm.cdf(-d_2)\
            - dCrit_Put*np.exp((db-dr)*dT)*st.norm.cdf(-d_1)
        
        #Condition to satisfy eq (24)
        dRHS= dEU_Put - (1-np.exp((db-dr)*dT)*st.norm.cdf(-d_1))*dCrit_Put/dq_1
        dLHS= iX - dCrit_Put
        
        #Condition value
        dError= abs(dLHS-dRHS)/iX

        #Finding next guess price:
        db_i= np.exp((db-dr)*dT)*st.norm.cdf(-d_1)*(1 - 1/dq_1)\
            + (1 - np.exp((db-dr)*dT)*st.norm.pdf(-d_1)/ds*np.sqrt(dT))/dq_1
            
        dCrit_Put= (iX + db_i*dCrit_Put - dRHS)/(1 + db_i)
        
    return dCrit_Put        

###########################################################
### dAme_Put= American_Put(dS, lq, dCrit_Put, iX, db, ds, dT, dr)
def American_Put(dS, lq, dCrit_Put, iX, db, ds, dT, dr):
    '''
    Computing the American Put option price.
    '''
    
    [dq_1, dq_2]= lq
    d_1=          (np.log(dCrit_Put/iX) + (db + 0.5*ds**2)*dT)/(ds*np.sqrt(dT))
    dA_1=         -(dCrit_Put/dq_1)*(1-np.exp((db-dr)*dT)*st.norm.cdf(-d_1))
    
    #eq (6)
    d_1S=    (np.log(dS/iX) + (db + 0.5*ds**2)*dT)/(ds*np.sqrt(dT))
    d_2S=    d_1S - ds*np.sqrt(dT)
    dEU_Put= iX*np.exp(-dr*dT)*st.norm.cdf(-d_2S)\
        - dS*np.exp((db-dr)*dT)*st.norm.cdf(-d_1S)
    
    #eq(25)
    if dS > dCrit_Put:
        dAme_Put= dEU_Put + dA_1*(dS/dCrit_Put)**dq_1
    if dS <= dCrit_Put:
        dAme_Put= iX - dS
        
    return [dEU_Put, dAme_Put]

###########################################################
### dAme_Call= American_Call(dS, lq, dCrit_Call, iX, db, ds, dT, dr)
def American_Call(dS, lq, dCrit_Call, iX, db, ds, dT, dr):
    '''
    Computing the American Call option price.
    '''
    
    [dq_1, dq_2]= lq
    d_1=          (np.log(dCrit_Call/iX) + (db + 0.5*ds**2)*dT)/(ds*np.sqrt(dT))
    dA_2=         (dCrit_Call/dq_2)*(1-np.exp((db-dr)*dT)*st.norm.cdf(d_1))
    
    #eq (5)
    d_1S=     (np.log(dS/iX) + (db + 0.5*ds**2)*dT)/(ds*np.sqrt(dT))
    d_2S=     d_1S - ds*np.sqrt(dT)
    dEU_Call= dS*np.exp((db-dr)*dT)*st.norm.cdf(d_1S)\
        - iX*np.exp(-dr*dT)*st.norm.cdf(d_2S)
    
    #eq (20)
    if dS < dCrit_Call:
        dAme_Call= dEU_Call + dA_2*(dS/dCrit_Call)**dq_2
    if dS > dCrit_Call:
        dAme_Call= dS - iX
    
    return [dS, dEU_Call, dAme_Call]

###########################################################
### lAme_Option_prices= Ame_Option_Price(vT, vr, vs, db, iX, vS)
def Ame_Option_Price(vT, vr, vs, db, iX, vS):
    ''' 
    Get list of American Option prices throughout the quadratic
    approximation method.
    '''
    lAme_Call= []
    lAme_Put=  []

    for i in range(len(vT)):
        dT= vT[i]
        dr= vr[i]
        ds= vs[i]
        
        lq_inf=      roots(db, dr, np.inf, ds)
        lSeedPrices= SeedPrices(lq_inf, db, dT, ds, iX)
        
        lq=         roots(db, dr, dT, ds)
        dCrit_Call= Critical_Call_Price(lSeedPrices, lq, iX, db, ds, dT, dr)
        dCrit_Put=  Critical_Put_Price(lSeedPrices, lq, iX, db, ds, dT, dr)
        
        for j in range(len(vS)):
            dS= vS[j]
            lAme_Call.append(American_Call(dS, lq, dCrit_Call, iX, db, ds, dT, dr))
            lAme_Put.append(American_Put(dS, lq, dCrit_Put, iX, db, ds, dT, dr))
    
    #Transpose list:
    lAme_Call= list(map(list, zip(*lAme_Call)))
    lAme_Put=  list(map(list, zip(*lAme_Put)))
    
    lS=        lAme_Call[0]
    lEU_Call=  lAme_Call[1]
    lEU_Put=   lAme_Put[0]
    lAme_Call= lAme_Call[2]
    lAme_Put=  lAme_Put[1]
    
    lEU_Call=  [round(x,2) for x in lEU_Call]
    lEU_Put=   [round(x,2) for x in lEU_Put]
    lAme_Call= [round(x,2) for x in lAme_Call]
    lAme_Put=  [round(x,2) for x in lAme_Put]
    
    return [lS, lEU_Call, lAme_Call, lEU_Put, lAme_Put]
    
###########################################################
### main()
def main():
    # Magic numbers

    db= -0.04                                 #Cost of Carry
    iX= 100                                   #Exercise price
    vT= np.array([0.25, 0.25, 0.25, 0.5])     #Time to maturity
    vr= np.array([0.08, 0.12, 0.08, 0.08])    #Risk free rate
    vs= np.array([0.2, 0.2, 0.4, 0.2])        #Volatity of underlying commodity
    vS= np.array([80, 90, 100, 110, 120])     #Commodity price

    
    # Initialisation
    [lS, lEU_Call, lAme_Call, lEU_Put, lAme_Put]= Ame_Option_Price(vT, vr, vs, db, iX, vS)
    
    # Output
    df= pd.DataFrame({'Commodity Price S': lS,
                      'European c(S, T)':  lEU_Call,
                      'American C(S, T)':  lAme_Call,
                      'European p(S, T)':  lEU_Put,
                      'American P(S, T)':  lAme_Put})
    
    table= df.to_latex(index=False)
    text_file = open("table.txt", "w")
    text_file.write(table)
    text_file.close()
    
    
###########################################################
### start main
if __name__ == "__main__":
    main()
