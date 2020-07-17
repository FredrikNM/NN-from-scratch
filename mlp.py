'''
    This pre-code is a nice starting point, but you can
    change it to fit your needs.
'''
import numpy as np


class mlp:
    def __init__(self, inputs, target, nhidden, momentum = 0):
        self.beta = 0.25
        self.eta = 0.1
        self.momentum = momentum
        self.N = np.shape(inputs)[0]
        print(np.shape(inputs))
        self.inputnodes = np.shape(inputs)[1]
        # self.inputs = inputs
        self.inputs = np.concatenate((inputs,-np.ones((self.N,1))), axis=1)
        self.weights1 = np.random.rand(np.shape(self.inputs)[1], nhidden)*np.sqrt(self.inputnodes)-(np.sqrt(self.inputnodes)/2)
        self.weights2 = np.random.rand(nhidden+1, np.shape(target)[1])*np.sqrt(self.inputnodes)-(np.sqrt(self.inputnodes)/2)
        # Weights are set at 1/sqrt(hidden nodes)

    def earlystopping(self, inputs, targets, valid, validtargets, rounds, treshold):
        tSSE = np.zeros(rounds)
        vSSE = np.zeros(rounds)
        # The Sum of squared error
        terror = np.zeros(rounds)
        verror = np.zeros(rounds)
        # % error from the n inputs
        iterations = rounds
        for i in range(rounds):
            tSSE[i], terror[i] = self.train(inputs, targets, iterations = 1)
            vSSE[i], verror[i], yk = self.forward(valid, validtargets)

            if i > 6 and verror[i] < treshold and all(np.greater_equal(verror[i], verror[-rounds+(i-1):-rounds+(i-4):-1])) == True or i > self.N and all(np.greater_equal(verror[i], verror[-rounds+(i-1):-rounds+(i- (self.N + int(self.N*0.1)) ):-1])) == True:
                #Stops the function at a selected treshold, or if the NN has run a % of the inputs without improving

                tSSE = tSSE[:i]
                vSSE = vSSE[:i]
                terror = terror[:i]
                verror = verror[:i]
                iterations = i
                break
            else:
                continue

        return terror, verror, iterations


    def train(self, inputs, targets, iterations):
        inputs = np.concatenate((inputs,-np.ones((self.N,1))), axis=1)

        for i in range(iterations):
            hi = np.dot(inputs,self.weights1)
            ai = 1/(1+np.exp(-self.beta*hi))
            ai = np.concatenate((ai,-np.ones((np.shape(ai)[0],1))), axis=1)
            hk = np.dot(ai,self.weights2)
            ak = 1/(1+np.exp(-self.beta*hk))

            deltao = (targets-ak)*ak*(1.0-ak)
            deltah = ai*(1.0-ai)*(np.dot(deltao,np.transpose(self.weights2)))

            updatew1 = np.zeros((np.shape(self.weights1)))
            updatew2 = np.zeros((np.shape(self.weights2)))

            updatew1 = self.eta*(np.dot(np.transpose(inputs),deltah[:,:-1])) + self.momentum*updatew1
            updatew2 = self.eta*(np.dot(np.transpose(ai),deltao)) + self.momentum*updatew2
            self.weights1 += updatew1
            self.weights2 += updatew2

        t = sum((targets)).astype(int)
        yk = sum((ak == ak.max(axis=1)[:,None]).astype(int))
        out = (ak == ak.max(axis=1)[:,None]).astype(int)
        SSE = 0.5*sum((t - yk)**2)
        wrong = sum(sum(abs(out - targets.astype(int))))/2
        errorate = float(wrong)/(np.shape(inputs)[0])        
        #Just calculating the errorrate, and the SSE

        return SSE, errorate


    def forward(self, inputs, targets):
        inputs = np.concatenate((inputs,-np.ones((np.shape(inputs)[0],1))), axis=1)
        hi = np.dot(inputs,self.weights1)
        ai = 1/(1+np.exp(-self.beta*hi))
        ai = np.concatenate((ai,-np.ones((np.shape(ai)[0],1))), axis=1)
        hk = np.dot(ai,self.weights2)
        ak = 1/(1+np.exp(-self.beta*hk))
        SSE = sum((ak-targets)**2)
        t = sum((targets)).astype(int)
        yk = sum((ak == ak.max(axis=1)[:,None]).astype(int))
        out = (ak == ak.max(axis=1)[:,None]).astype(int)
        SSE = 0.5*sum((t - yk)**2)
        wrong = sum(sum(abs(out - targets.astype(int))))/2
        errorate = float(wrong)/(np.shape(inputs)[0])

        return SSE, errorate, out


    def confusion(self, out, targets):
        mat = np.zeros((np.shape(targets)[1],np.shape(targets)[1]))

        for i in range(np.shape(out)[0]):
            posout = int(np.where(out[i] == 1)[0])
            postest = int(np.where(targets[i] == 1)[0])

            if all(out[i] == targets[i].astype(int)):
                mat[posout,posout] += 1
            else:
                mat[posout, postest] += 1
        return mat