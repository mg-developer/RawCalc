
#include <stdlib.h>
#include <stdio.h>
#include <dlfcn.h>
#include <iostream>
#include <iomanip>

int main(int argc, char *argv[]) {

    void *handle;
    typedef int (*IVoidFn)(void);
    char *error;

    IVoidFn ModuleSetup;

    handle = dlopen ("./libRawCalc.so", RTLD_NOW);
    if (!handle) {
        std::cerr << dlerror() << std::endl;
        exit(1);
    }

    std::cout << "Init - ok" << std::endl;
    ModuleSetup = (IVoidFn)dlsym(handle, "ModuleSetup");

    std::cout << "ptr:" << std::hex << (void*)ModuleSetup << std::endl;
    if ((error = dlerror()) != NULL)  {
        std::cerr << dlerror() << std::endl;
        exit(1);
    }

    std::cout << "Fn - ok" << std::endl;
    (*ModuleSetup)();

    dlclose(handle);
    return 0;
}


