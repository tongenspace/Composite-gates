import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import scipy
from matplotlib.pyplot import figure
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import (MultipleLocator,
                               FormatStrFormatter,
                               AutoMinorLocator)


import os
os.environ["PATH"] += ':/usr/local/texlive/2015/bin/x86_64-darwin'
#Function that creates the Hamiltonina
N = 6
global losata
losata = []
na = 5


def infidelity_operator(parameters,kude = [14],operatora = 'X',name = "",N = 7):
  list_of_differences = []
  for na in kude:
      mia = []
      for j in range(2,22):
        operator = create_gate(operatora,j)
        plotnato,y = plot_results(parameters,False,N,j,operator = True, G = operator,gate = operatora)
        mia.append(1- plotnato)    
      x = np.arange(2,len(mia)+2)
      
      plt.plot(x,np.array(mia),'g',label=f"n = {na}")
      
      plt.legend()
      plt.xlim(2,21)
      print(mia)
      #plt.ylim(10e-4,5*10e-2)
      plt.xlabel(f"$n_j$",fontsize = 14)
      plt.ylabel(r"$Infidelity_{n_j}$",fontsize = 14)
      plt.yscale('log')
      plt.savefig(f"{name}.pdf",format="pdf")
      plt.show()

def infidelity(parameters,kude = [14],name = ""):
  if len(parameters)!=14:
    params = parameters
  else:
     params = parameters[13]
  list_of_differences = []
  for na in kude:
      mia = []
      for j in range(3,19):
        plotnato,y = plot_results(params,False,len(params)//3,j,infidel = True)
        mia.append(plotnato)    

      x = np.arange(3,len(mia)+3)
      #figure(figsize=(4, 3), dpi=120)
      #fig, ax = plt.subplots()
      plt.plot(x,np.abs(np.array(mia)),'g',label=f"n = {na}")
      plt.yscale('log')
      
      #ax.tick_params(labelsize=11)
      #ax.rc('text', usetex=True)
      #ax.rc('font', family='serif')
      plt.legend()
      plt.xlim(3,18)
      print(mia)
      #plt.ylim(0.5 - 0.4991565059437427,0.5-0.4991687372725388)
      plt.xlabel(f"$n_j$",fontsize = 14)
      plt.ylabel(r"$\Delta P$",fontsize = 14)
      
      # after plotting the data, format the labels

      #ax.set_yscale('log')
      #.yaxis.set_major_formatter(FormatStrFormatter('%.5f'))

      plt.savefig(f"{name}.pdf",format="pdf")
      
      plt.show()

def diferentiate_operators(parameters,kude = [14],N = 6, operatora = 'X',name = ""):
  list_of_differences = []
  for na in kude:
      mia = []
      for j in range(3,22):
        operator = create_gate(operatora,j)
        plotnato,y = plot_results(parameters,False,N,j,operator = True, G = operator,gate = operatora)
        mia.append(y)      
      list_of_differences.append(mia)
  
  diff = []
  for m in range(len(list_of_differences[0])):
      diff.append(np.sum(np.abs(np.array(list_of_differences[0][kude[0] - 3])[14:27]-np.array(list_of_differences[0][m])[14:27])))
  x = np.arange(3,len(diff)+3)

  plt.plot(x,np.array(diff),'rx',label=f"n = {kude[0]}")
  
  plt.legend()
  plt.xlim(3,21)
  print(diff)
  #plt.ylim(0,10e-3)
  plt.xlabel(f"$n_j$",fontsize = 14)
  plt.ylabel(r"$\delta F_{n_j}$",fontsize = 14)
  plt.title(f'Fidelity difference of the {operatora} gate for different N')
  plt.yscale('log')
  plt.savefig(f"{name}.pdf",format="pdf")
  plt.show()

def diferentiate(parameters,kude = [14],name = ""):
  list_of_differences = []
  for na in kude:
      mia = []
      for j in range(3,15):
        plotnato,y = plot_results(parameters[na-3],False,N,j)
        mia.append(y)      
      list_of_differences.append(mia)

  for j in range(len(list_of_differences)):
    diff = []
    for m in range(len(list_of_differences[j])):
      diff.append(np.sum(np.abs(np.array(list_of_differences[j][kude[j] - 3])[14:27]-np.array(list_of_differences[j][m])[14:27])))
    x = np.arange(3,len(diff)+3)
    #figure(figsize=(4, 3), dpi=120)
    plt.tick_params(labelsize=11)
    plt.plot(x,np.array(diff),'rx',label=f"n = {kude[j]}")
    
    plt.legend()
    plt.xlim(3,14)
    print(diff)
    #plt.ylim(0,10e-3)
    plt.xlabel(f"$n_j$",fontsize = 14)
    plt.ylabel(r"$\delta P_{n_j}$",fontsize = 14)
    plt.yscale('log')
    plt.savefig(f"{name}.pdf",format="pdf")
    plt.show()

def rabbipart(Rabbi,epsilon,phase,n): 
  return np.sqrt(n)*(Rabbi*(1+epsilon)*np.exp(1j*phase))/2

def Hamiltonian(Rabbi,phase,epsilon,detunning=-2*np.pi*0.3, n = 5):
   H = np.zeros((n,n)).astype(np.complex128)
   for j in range(n):
     for i in range(n):
      if i == j and i >= 2:
       H[i,j] = ((i-1)**2+(i-1))/2*detunning
      elif i == j-1:
        H[i,j] = rabbipart(Rabbi,epsilon,phase,j)
      elif i == j+1:
        H[i,j] = np.conj(rabbipart(Rabbi,epsilon,phase,i))
   return H

def Create_U_n(t,N,phi,Rabbi,epsilon,detun =-2*np.pi*0.3,  mui = 3):
  Propegator = np.diag(np.ones(mui)).astype(np.complex128)
  for j in range(N):
    hamilton = Hamiltonian(Rabbi[j],phi[j],epsilon,detunning = detun,n = mui)
    Propegator = np.matmul(scipy.linalg.expm(-1j*t[j]*hamilton),Propegator)
  return Propegator

def Loss_Function(Prob,N,phase,Rabbi,time,sigma = 0.25,n = 5):
  eps = np.arange(-0.31,0.31,0.1)
  U = []
  #eps = np.array([0.])
  for e in eps:
    U.append(Create_U_n(time,N,phase,Rabbi,e, mui = n))

  loss = 0.
  gaus = 1/sigma/np.sqrt(2*np.pi)*np.exp(-1/2*(eps/sigma)**2)
  for m in range(len(eps)):
   for j in range(2):
    loss += np.abs(Prob[j]-np.conj(U[m][j,0])*U[m][j,0])*gaus[m]
  loss += np.sum(np.abs(Rabbi)*time)/4
  return loss 

def function_to_minimse(Rabbiphase,*args):
  global losata
  N = len(Rabbiphase)//3
  Prob = [args[0][0],args[0][1]]
  n = args[0][2]
  Rabbi,phase,time = Rabbiphase[:len(Rabbiphase)//3],Rabbiphase[len(Rabbiphase)//3:2*len(Rabbiphase)//3],Rabbiphase[2*len(Rabbiphase)//3:]
  losa = Loss_Function(Prob,N,phase,Rabbi,time, n = n)
  losata.append(losa)
  return losa

Bound = [(-10.,10.) for j in range(2*N)]
for j in range(N):
  Bound.append((0.5,5.))

def plot_results(res_pop_trasfer,pop_transfer,N,n,operator = False, G = np.diag(np.ones(3)),gate = 'X',infidel = False,name = ""):
    global P2,e_sults_x,e_pop
    epsilon = np.arange(-0.2,0.2,0.01)
    y,y1,y2 = [],[],[]
    #Rabbi,phase,time = res_pop_trasfer[:len(res_pop_trasfer)//2],res_pop_trasfer[len(res_pop_trasfer)//2:],np.ones(len(res_pop_trasfer)//2)*0.5
    Rabbi,phase,time = res_pop_trasfer[:len(res_pop_trasfer)//3],res_pop_trasfer[len(res_pop_trasfer)//3:2*len(res_pop_trasfer)//3],res_pop_trasfer[2*len(res_pop_trasfer)//3:]
    time_single = np.sum(time)
    if gate == 'X':
     Rabbi_single = np.pi/time_single
    else:
      Rabbi_single = np.pi/time_single/2
    for e in epsilon:
        U = Create_U_n(time,N,phase,Rabbi,e, mui = n)
        U_single = Create_U_n([time_single],1,[0.],[Rabbi_single],e, mui = n)

        if operator:
         res = -optimal_trace(U,G,n = 2)+1
         y1.append(-optimal_trace(U_single,G,n = 2)+1)
         y.append(res) 
        else:
         res = (np.conj(U[1,0])*U[1,0]).real #np.sqrt(np.einsum("i,ij,j->",[0.,1.,0.],U,[1.,0.,0.])*np.conj(np.einsum("i,ij,j->",[0.,1.,0.],U,[1.,0.,0.]))).real
         resdr = (np.conj(U_single[1,0])*U_single[1,0]).real

         if n>2:
          resoshe = np.sum(np.conj(U[:,0])*U[:,0]).real
         if infidel:
            res = 1-res
            resdr = 1- resdr
            resoshe = 1-resoshe
         y.append(res)
         y1.append(resdr)
         if n>2:
           y2.append(resoshe)

    if   pop_transfer:
            #figure(figsize=(4, 3), dpi=120)
            plt.tick_params(labelsize=11)
            print("Area = ",np.sum(np.abs(Rabbi)*time))
            #print(y[len(y)//2])
            #plt.plot(epsilon,y1,label="P1")
            if operator:
              plt.plot(epsilon[5:36],1-np.array(y[5:36]),label=f"CP")#,epsilon,y1,epsilon,y2)
              if gate == 'sqrtX':
                print('H')
                plt.plot(epsilon[5:36],1-np.array(y1[5:36]),label=r"$\frac{\pi}{2}$ pulse")
              elif gate == 'X':
               plt.plot(epsilon[5:36],1-np.array(y1[5:36]),label=r"$\pi$-pulse")
              if gate == 'X':
               plt.plot(epsilon[5:36],np.array(e_sults_x[6:37]),label=f"DRAG")
              if gate == 'sqrtX':
                plt.title(r'$\sqrt{X}$ gate infidelity')
              else:
                plt.title(f'{gate} gate infidelity')
              plt.yscale('log')
              #print(U)
            else:
             if gate == 'sqrtX' :
              plt.plot(epsilon[5:36],np.abs(0.5-np.array(y1[5:36])),label=r"$\frac{\pi}{2}$ pulse")
              plt.plot(epsilon[5:36],np.abs(0.5-np.array(y[5:36])),label="CP")
              plt.title('Half population Transfer Error')
             else:
              plt.plot(epsilon[5:36],1-np.array(y1[5:36]),label=r"$\pi$-pulse")
              plt.plot(epsilon[5:36],1-np.array(e_pop[6:37]),label=f"DRAG")
              plt.title('Population Transfer Error')
              plt.plot(epsilon[5:36],1-np.array(y[5:36]),label="CP")
             
             plt.yscale('log')
             #if n>2:
             #  plt.plot(epsilon,y2,label="Leakage")
            #plt.plot(epsilon,P2,label="P2-Boyan")
            plt.legend()
            plt.xlim(-1,1)
            if operator:
               plt.ylim(0.,0.05)
               plt.xlim(-0.15,0.15)
            else:
               #plt.ylim(0,0.1)
               plt.xlim(-0.15,0.15)
            plt.xlabel(r"$\epsilon$",fontsize = 14)
            if operator:
                plt.ylabel(r"$Infidelity$",fontsize = 14)
                print('average = ',np.average(y[len(y)//2-4:len(y)//2+5]))
            else:
                plt.ylabel(r"Populations Error",fontsize = 14)
            print([(epsilon[j],y[j]) for j in range(len(epsilon))])
            #print([(epsilon[j],y2[j]) for j in range(len(epsilon))])
            #print(np.max(np.array(y)))
            plt.savefig(f"{name}.pdf",format="pdf")
            plt.show()
    return y[len(y)//2],np.array(y),

#[ 4.54870693e-01 -2.05263543e-01  6.89397427e-01  6.76439438e-01 7.27384072e-08 -1.37316238e-11  1.20528542e-01  2.16373413e+00 -1.87872416e+00  4.22449923e-02  3.45823738e-01  1.51462149e+00 2.57754692e+00  1.28171778e+00  1.64646708e+00  2.05904734e+00 2.38587543e+00  1.26537406e+00]
def create_gate(which,levels):          # Function to create gates with which to compare the generated Propagator
    array = np.diag(np.ones(levels)).astype(np.complex128)
    if which == 'X':
      array[0][0] = 0.
      array[0][1] = 1.
      array[1][0] = 1.
      array[1][1] = 0.
    elif which == 'T':
        array[1][1] = np.exp(1j*np.pi/4)
    elif which == 'H':
      array[0][0] = 1/np.sqrt(2)
      array[0][1] = 1./np.sqrt(2)
      array[1][0] = 1/np.sqrt(2)
      array[1][1] = -1/np.sqrt(2)
    elif which == 'sqrtX':
      array[0][0] = 1/2*(1j+1)
      array[0][1] = 1/2*(-1j+1)
      array[1][0] = 1/2*(-1j+1)
      array[1][1] = 1/2*(1j+1)
    return array

def bez_proektor(U,G):
  na = len(U)
  M = np.matmul(np.transpose(np.conj(G)),U)
  Fidelity = 1/(na*(na+1))*(np.trace(np.matmul(M,np.transpose(np.conj(M))))+np.square(np.abs(np.trace(M))))
  return 1-Fidelity.real

def optimal_trace(U,G,n = 2):            # Fidelity function taken from https://arxiv.org/pdf/quant-ph/0701138.pdf equation (3)
  projector = np.zeros_like(U)
  projector[:n,:n] = np.diag(np.ones(n))

  M = np.matmul(projector,np.transpose(np.conj(G)))
  M = np.matmul(M,U)
  M = np.matmul(M,projector)
  Fidelity = 1/(n*(n+1))*(np.trace(np.matmul(M,np.transpose(np.conj(M))))+np.square(np.abs(np.trace(M))))
  return 1-Fidelity.real
def Loss_Function_operator(G,N,phase,Rabbi,time,sigma = 0.25,n = 5):
  eps = np.arange(-0.30,0.31,0.1)
  U = []
  #eps = np.array([0.])
  for e in eps:
    U.append(Create_U_n(time,N,phase,Rabbi,e, mui = n))

  loss = 0.
  gaus = 1/sigma/np.sqrt(2*np.pi)*np.exp(-1/2*(eps/sigma)**2)
  for m in range(len(eps)):
    loss += optimal_trace(U[m],G,n = 2)*gaus[m]
  loss += np.sum(np.abs(Rabbi)*time)/10
  return loss 

def function_to_minimse_for_operator(Rabbiphase,*args):
  N = len(Rabbiphase)//3
  G = args[0][0]
  n = args[0][1]
  #Rabbi,phase,time = Rabbiphase[:len(Rabbiphase)//2],Rabbiphase[len(Rabbiphase)//2:],np.ones(len(Rabbiphase)//2)*0.5
  Rabbi,phase,time = Rabbiphase[:len(Rabbiphase)//3],Rabbiphase[len(Rabbiphase)//3:2*len(Rabbiphase)//3],Rabbiphase[2*len(Rabbiphase)//3:]
  losh = Loss_Function_operator(G,N,phase,Rabbi,time,sigma = 0.25,n = n)
  return losh


def plot_results11(res_pop_trasfer,resa_drug,pop_transfer,N,n,operator = False, G = np.diag(np.ones(3)),gate = 'X',name = "",delta_compar = False):
    global P2
    if delta_compar:
      epsilon = np.arange(2/3,1.34,0.01) * 2*np.pi*0.3
    else:
      epsilon = np.arange(-1.00,1.01,0.01)

    y,y1,y2,yU,yU1 = [],[],[],[],[]
    #Rabbi,phase,time = res_pop_trasfer[:len(res_pop_trasfer)//2],res_pop_trasfer[len(res_pop_trasfer)//2:],np.ones(len(res_pop_trasfer)//2)*0.5
    Rabbi,phase,time = res_pop_trasfer[:len(res_pop_trasfer)//3],res_pop_trasfer[len(res_pop_trasfer)//3:2*len(res_pop_trasfer)//3],res_pop_trasfer[2*len(res_pop_trasfer)//3:]
    Rabbi1,phase1,time1 = resa_drug[:len(resa_drug)//3],resa_drug[len(resa_drug)//3:2*len(resa_drug)//3],resa_drug[2*len(resa_drug)//3:]
    
    for e in epsilon:
        if delta_compar:
         U = Create_U_n(time,N,phase,Rabbi,0., mui = n,detun = e)
         drugU = Create_U_n(time1,1,phase1,Rabbi1,0., mui = n,detun = e)
        else:
          U = Create_U_n(time,N,phase,Rabbi,e, mui = n)
          drugU = Create_U_n(time1,1,phase1,Rabbi1,e, mui = n)
        resdrU = 0
        if operator:
         res = -optimal_trace(U,G,n = 2)+1
         resU = -optimal_trace(drugU,G,n = 2)+1
         y.append(res) 
         y1.append(resU) 
        else:
         res = (np.conj(U[1,0])*U[1,0]).real #np.sqrt(np.einsum("i,ij,j->",[0.,1.,0.],U,[1.,0.,0.])*np.conj(np.einsum("i,ij,j->",[0.,1.,0.],U,[1.,0.,0.]))).real
         resdr = (np.conj(U[0,0])*U[0,0]).real
         
         resdrU  = (np.conj(drugU[1,0])*drugU[1,0]).real
         resdrU1  = (np.conj(drugU[0,0])*drugU[0,0]).real
         y.append(res)
         y1.append(resdr)
         yU1.append(resdrU)
         y2.append(resdrU1)

    if   pop_transfer:
            #figure(figsize=(4, 3), dpi=120)
            plt.tick_params(labelsize=11)
            print("Area = ",np.sum(np.abs(Rabbi)*time))

            fig, ax = plt.subplots()
            #print(y[len(y)//2])
            #plt.plot(epsilon,y1,label="P1")

            if operator:
              ax.plot(epsilon,y,label=r"composite $\sqrt{X}$ gate")#,epsilon,y1,epsilon,y2)
              if gate == 'X' or gate == 'H' or gate == 'sqrtX':
               ax.plot(epsilon,y1,'--',label=r"$\frac{\pi}{2}$ pulse ")#,epsilon,y1,epsilon,y2)
              
              #print(U)
              ax.legend()
            else:
             left, bottom, width, height = 0.7, 0.15, 0.3, 0.25
             inset_ax = fig.add_axes([left, bottom, width, height])
             ax.plot(epsilon,y,label=r"$P_1$ composite")
             ax.plot(epsilon,yU1,'y--',label=r"$P_1$ $\pi$ puplse")
             ax.plot(epsilon,y1,label=r"$P_2$ composite")
             ax.plot(epsilon,y2,'r--',label=r"$P_2$ $\pi$ puplse")
             inset_ax.plot(epsilon, 1- np.array(yU1) - np.array(y2),'g--')
             inset_ax.plot(epsilon, 1- np.array(y) - np.array(y1),'g')
             inset_ax.set_yscale('log')
             inset_ax.set_ylim(10**-4,5*10**-2)
             print([(epsilon[j],1- np.array(yU1)[j] - np.array(y2)[j]) for j in range(len(epsilon))])
             print([(epsilon[j],1- np.array(y)[j] - np.array(y1)[j]) for j in range(len(epsilon))],"\n")

            #plt.plot(epsilon,P2,label="P2-Boyan")
            
            ax.set_xlim(-1,1)
            if operator:
               ax.set_ylim(0.95,1)
               ax.set_xlim(-0.2,0.2)
            else:
               ax.set_ylim(0,1)
               print([(epsilon[j],yU1[j]) for j in range(len(epsilon))])
            if delta_compar:
              ax.set_xlabel(r"$\delta$",fontsize = 14)
            else: 
             ax.set_xlabel(r"$\epsilon$",fontsize = 14)
            if operator:
                ax.set_ylabel(r"$Fidelity$",fontsize = 14)
                print('average = ',np.average(y[len(y)//2-4:len(y)//2+5]))
                if gate == 'X':
                 print([(epsilon[j],y1[j]) for j in range(len(epsilon))])
            else:
                ax.set_ylabel(r"$P$",fontsize = 14)
            print([(epsilon[j],y[j]) for j in range(len(epsilon))])
            

            #print([(epsilon[j],y2[j]) for j in range(len(epsilon))])
            #print(np.max(np.array(y)))
            plt.legend()
            plt.savefig(f"{name}.pdf",format="pdf")
            plt.show()
    return y[len(y)//2],np.array(y)


def plot_detuning(res_pop_trasfer,pop_transfer,N,n,operator = False, G = np.diag(np.ones(3)),gate = 'X',name = "",delta_compar = True):

    if delta_compar:
      epsilon = np.arange(1.8849555921538759-0.5,1.8849555921538759+0.5,0.01)

    y,y1,y2 = [], [], []
    Rabbi,phase,time = res_pop_trasfer[:len(res_pop_trasfer)//3],res_pop_trasfer[len(res_pop_trasfer)//3:2*len(res_pop_trasfer)//3],res_pop_trasfer[2*len(res_pop_trasfer)//3:]
    
    for e in epsilon:

        U = Create_U_n(time,N,phase,Rabbi,0., mui = n,detun = e)

        resdrU = 0
        if operator:
         res = -optimal_trace(U,G,n = 2)+1

         y.append(res) 

        else:
         res = (np.conj(U[1,0])*U[1,0]).real 
         resdr = (np.conj(U[0,0])*U[0,0]).real
         
         y.append(res)
         y1.append(resdr)


    if   pop_transfer:
            #figure(figsize=(4, 3), dpi=120)
            plt.tick_params(labelsize=11)
            print("Area = ",np.sum(np.abs(Rabbi)*time))


            if operator:
              plt.plot(epsilon,y,label=r"composite $X$ gate")
              
              #print(U)
              plt.legend()
            else:
             plt.plot(epsilon,np.array(y),label=r"$P_1$ composite")
             print([(epsilon[j],1- np.array(y)[j] - np.array(y1)[j]) for j in range(len(epsilon))],"\n")
            
            plt.title(f"{gate} Fidelity")
            #plt.ylim(0,1)

            if delta_compar:
              plt.xlabel(r"$\delta$[GHz]",fontsize = 14)

            if operator:
                plt.ylabel(r"$Fidelity$",fontsize = 14)
                print('average = ',np.average(y[len(y)//2-4:len(y)//2+5]))
            else:
                plt.ylabel(r"$P$",fontsize = 14)
            print([(epsilon[j],y[j]) for j in range(len(epsilon))])
            

            #print([(epsilon[j],y2[j]) for j in range(len(epsilon))])
            print(np.max(np.array(y)))
            plt.legend()
            plt.savefig(f"{name}.pdf",format="pdf")
            plt.show()
    return y[len(y)//2],np.array(y)

# DRAG results for X gate Infidelity
e_sults_x = [0.06619686461574559,
0.060160107707515786,
 0.05438703422935143,
 0.04888286140443099,
 0.0436526314987018,
 0.03870101827041539,
 0.034032435870484656,
 0.02965102240133355,
 0.025560607614226294,
 0.021764800090606218,
 0.01826684056382355,
 0.015069685909395947,
 0.0121760488905025,
 0.009588287329663547,
 0.007308475439061324,
 0.0053383890261299305,
 0.003679492698846043,
 0.0023329743971014505,
 0.0012996889271759127,
 0.000580197149408157,
 0.00017476719928866125,
 8.335288344918368e-05,
 0.0003056109388145378,
 0.0008408991012712752,
 0.0016882772840512983,
 0.0028465082523345675,
 0.004314081392116975,
 0.006089137397991662,
 0.008169584862278922,
 0.010553023208910162,
 0.01323677550518676,
 0.0162178809572342,
 0.019493107310649016,
 0.023058952018668144,
 0.02691168770797092,
 0.031047196565494928,
 0.03546122635019233,
 0.04014923710037266,
 0.045106441326354996,
 0.0503278112403307]

# DRAG results for Population transfer   
e_pop = np.array([0.9007307248950471,
 0.9097863898095805,
 0.9184465389042494,
0.9267033348200501,
 0.9345492099799722,
0.9419771884329926,
 0.9489805869519576,
 0.955553225765173,
 0.9616893400538644,
 0.9673835454461887,
0.9726309972176733,
0.977427233419414,
 0.9817681832597898,
 0.985650319848796,
0.9890705806823279,
0.992026129492091,
 0.994514969675631,
 0.9965351804939186,
 0.9980855648088597,
 0.9991652520914096,
 0.9997738294013958,
 0.9999113774570382,
 0.9995784055474578,
 0.9987758788167433,
 0.9975051924474595,
 0.9957682387198272,
 0.9935672575125717,
 0.9909050281871671,
 0.9877846708349365,
 0.9842098829536905,
 0.9801845286027187,
 0.975713191643909,
 0.970800615341415,
0.9654521643057407,
 0.9596733209244851,
0.9534702322169977,
0.9468494779469152,
0.9398176523438273,
0.9323820419364282,
0.9245501719933611,])