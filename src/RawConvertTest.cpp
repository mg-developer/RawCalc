
#include <stdlib.h>
#include <stdio.h>
#include <dlfcn.h>
#include <iostream>
#include <iomanip>
#include <fstream>

#include "RawStructs.h"

unsigned char *loadFileMem(const char* file, int &s) {
    unsigned char * buffer = NULL;
    int length;
    std::ifstream is(file, std::ifstream::binary);
    if (is) {
        is.seekg (0, is.end);
        length = is.tellg();
        is.seekg (0, is.beg);
        buffer = new unsigned char [length];
        is.read ((char*)buffer,length);
        is.close();
    }
    s = length;
    return buffer;
}

int main(int argc, char *argv[]) {

    void *handle;
    typedef int (*IMeta)(metadata);
    typedef int (*IStr)(char *);
    typedef int (*IStrMlvDng)(char *, mlvBlock, unsigned char *, int);

    char *error;

    IMeta ModuleSetup;
    IStr AssignFileToProcess;
    IStrMlvDng ProcessDataAndSave;

    handle = dlopen ("./libRawCalc.so", RTLD_NOW);
    if (!handle) {
        std::cerr << dlerror() << std::endl;
        exit(1);
    }


    ModuleSetup = (IMeta)dlsym(handle, "ModuleSetup");
    AssignFileToProcess = (IStr)dlsym(handle, "AssignFileToProcess");
    ProcessDataAndSave = (IStrMlvDng)dlsym(handle, "ProcessDataAndSave");

    std::cout << "Load - ok" << std::endl;

    std::cout << "ptr:" << std::hex << (void*)ModuleSetup << std::endl;
    if ((error = dlerror()) != NULL)  {
        std::cerr << dlerror() << std::endl;
        exit(1);
    }

    std::cout << std::dec;
    int size;
    metadata *md = (metadata*)loadFileMem("../test/metadata.out",size); 
    std::cout << "metadata.out - loaded [" << size << "]" << std::endl;
    int sizedng = 0;
    unsigned char *dng = loadFileMem("../test/dng.out", sizedng); 
    std::cout << "dng.out - loaded [" << sizedng << "]" << std::endl;
    mlvBlock *mlv = (mlvBlock*)loadFileMem("../test/mlvblock.out",size); 
    std::cout << "mlvblock.out - loaded [" << size << "]" << std::endl;


    //1
    (*ModuleSetup)(*md);
    //2
    (*AssignFileToProcess)("../test/M27-1006.MLV");
    //3
    (*ProcessDataAndSave)("../test/M27-1006-1.dng", *mlv, dng, sizedng);


    dlclose(handle);


    delete [] (unsigned char*)md;
    delete [] dng;
    delete [] (unsigned char*)mlv;

    return 0;
}


