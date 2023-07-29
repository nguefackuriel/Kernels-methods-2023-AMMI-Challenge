from tqdm import tqdm
import pandas as pd
from collections import deque

############################################### Data Loader #######################################
class DataLoader():
    
    def __init__(self, fname):
        self.X = pd.read_csv(fname)['seq']
        self.data = self.X
        self.kmer_set = {}
        self.neigborhoods = {}
        self.alph = "GATC"
        self.precomputed = {}
        
    def spectrum_preprocess(self, k):
        """ 
        This method generates a k-mer embedding for each DNA sequence in the dataset. 
        A k-mer is a substring of length k extracted from the sequence. 
        The method counts the occurrences of each k-mer in each sequence and stores the count as an embedding, 
        representing the frequency of each k-mer in the sequence.
        """

        n = self.X.shape[0]
        d = len(self.X[0])
        embedding = [{} for x in self.X]
        print("Generating kmer embedding")
        for i,x in enumerate(tqdm(self.X)):
            for j in range(d - k + 1):
                kmer = x[j: j + k]
                if kmer in embedding[i]:
                    embedding[i][kmer] += 1
                else:
                    embedding[i][kmer] = 1
        self.data = embedding
        
    def populate_kmer_set(self, k):
        """ 
        This method populates the k-mer set, which stores unique k-mers present in the dataset. 
        It assigns a unique index to each k-mer, creating a mapping between k-mer strings and their corresponding indices.
        """
        d = len(self.X[0])
        idx = 0
        print("Populating kmer set")
        for x in tqdm(self.X):
            for j in range(d - k + 1):
                kmer = x[j: j + k]
                if kmer not in self.kmer_set:
                    self.kmer_set[kmer] = idx
                    idx +=1  
            
    def mismatch_preprocess(self, k, m):
        """ 
        This method generates a mismatch embedding for each DNA sequence in the dataset. A mismatch is defined as a 
        nucleotide in a k-mer that is allowed to differ from the original nucleotide. The method finds the mismatched 
        k-mers within a specified mismatch distance (m) from each k-mer in the sequence. It then stores the indices of 
        these mismatched k-mers in the embedding, representing their frequency in the sequence.
        """
        n = self.X.shape[0]
        d = len(self.X[0])
        embedding = [{} for x in self.X]
        print("Generating mismatch embedding")
        for i,x in enumerate(tqdm(self.X)):
            for j in range(d - k + 1):
                kmer = x[j: j + k]
                if kmer not in self.precomputed:
                    Mneighborhood = self.m_neighborhood(kmer, m)
                    self.precomputed[kmer] = [self.kmer_set[neighbor] for neighbor in Mneighborhood if neighbor in self.kmer_set]
                    
                for idx in self.precomputed[kmer]:
                    if idx in embedding[i]:
                        embedding[i][idx] += 1
                    else:
                        embedding[i][idx] = 1
        self.data = embedding
            
    def m_neighborhood(self, kmer, m):
        """ 
        This method computes the mismatch neighborhood of a given k-mer. 
        It generates all possible mismatched k-mers within the specified mismatch distance (m) from the input k-mer.
        """
        mismatch_list = deque([(0, "")])
        for letter in kmer:
            num_candidates = len(mismatch_list)
            for i in range(num_candidates):
                mismatches, candidate = mismatch_list.popleft()
                if mismatches < m :
                    for a in self.alph:
                        if a == letter :
                            mismatch_list.append((mismatches, candidate + a))
                        else:
                            mismatch_list.append((mismatches + 1, candidate + a))
                if mismatches == m:
                    mismatch_list.append((mismatches, candidate + letter))
        return [candidate for mismatches, candidate in mismatch_list]