class embedded:
    def __init__(self):
        super(embedded, self).__init__()
    
    def origial(self,x):
        return x

    def gender(self,x):
    # 0: female, 1:male
        gen = []
        for i in range(len(x)):
            if len(set(x[i][-687:]))==1:
                gen.append(0)
            else:
                gen.append(1) 
        return np.array(gen).reshape(-1, 1)
    
    def pca(self,x):
        pca=decomposition.PCA(n_components=40)
        reduced_x = pca.fit_transform(x)
        return reduced_x
    
    def nmf(self,x):
        nmf = decomposition.NMF(n_components=200, init='nndsvda', tol=5e-3)
        reduced_x = nmf.fit_transform(np.c_[x[:,:-687],self.gender(x)])
        return reduced_x
        
        
    def emb(self,x):  
        gen = self.gender(x)
        pcs = self.pca(x)
        data = np.c_[pcs,gen]
        return data
        
    def emb_x(self,x):  
        gen = self.gender(x)
        data = np.c_[x,gen]
        return data 
