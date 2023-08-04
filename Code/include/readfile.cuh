#ifndef FILE_READER__
#define FILE_READER__

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

template<typename T>
T *readFile(string filePath, vector<int> &label, int &mat_rows, int &mat_cols)
{
    vector <vector <T> > data;
    ifstream infile(filePath);
    
    while (infile)
    {
        string s;
        if (!getline(infile, s)) break;

        istringstream ss(s);
        vector <T> record;

        int count = 0;

        while (ss)
        {
            string s;
            if (!getline(ss, s, ',')) break;
            //if (!getline(ss, s, '	')) break;
            if(count == 0)
            {
                int tmp = stoi(s);
                label.push_back(tmp);
            }
            else
            {
                T tmp = atof(s.c_str());
                record.push_back(tmp);
            }
            count++;
        }
        
        data.push_back(record);
    }
    
    if (!infile.eof())
    {
        cerr << "File eof error!\n";
    }

    infile.close();
    mat_rows = data.size();
    mat_cols = data.at(0).size();
    
    T *mat = new T[mat_rows*mat_cols];
    for (int i = 0; i < mat_rows; i++)
    {
        for (int j = 0; j < mat_cols; j++)
        {
            mat[j + i*mat_cols] = data[i][j];
        } 
    }
    return mat;
}




template<typename T>
T *read_unlabeled_File(string filePath, int &mat_rows, int &mat_cols)
{
    vector <vector <T> > data;
    ifstream infile(filePath);
    
    while (infile)
    {
        string s;
        if (!getline(infile, s)) break;

        istringstream ss(s);
        vector <T> record;


        while (ss)
        {
            string s;
            if (!getline(ss, s, ',')) break;
            T tmp = atof(s.c_str());
            record.push_back(tmp);
        }
        
        data.push_back(record);
    }
    
    if (!infile.eof())
    {
        cerr << "File eof error!\n";
    }

    infile.close();
    mat_rows = data.size();
    mat_cols = data.at(0).size();
    
    T *mat = new T[mat_rows*mat_cols];
    for (int i = 0; i < mat_rows; i++)
    {
        for (int j = 0; j < mat_cols; j++)
        {
            mat[j + i*mat_cols] = data[i][j];
        } 
    }
    return mat;
}



template<typename T>
void writeFile(string filePath, T *data, const int dataSize)
{
    ofstream fout(filePath);
    for (int i = 0; i < dataSize; i++)
    {
        fout<<data[i]<<",";
    }
    fout.close();
}

template<typename T>
void appendFile(T *data, const int dataSize, string file_path)
{
    fstream f;
    f.open(file_path, ios::out|ios::app);
    for (int i = 0; i < dataSize; i++)
    {
        f<<data[i]<<",";
    }
    f<<endl;
    f.close();
}



#endif
