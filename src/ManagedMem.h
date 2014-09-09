#ifndef _MANAGED_MEM_H
#define _MANAGED_MEM_H
#include <cuda_runtime.h>
#include <algorithm>


template<typename T>
class ManagedIterator : public std::iterator<std::bidirectional_iterator_tag, T> {
public:
    ManagedIterator(T *ptr = NULL) : mptr(ptr) {
    }

    bool operator==(ManagedIterator const& rhs) const {
        return (mptr==rhs.mptr);
    }

    bool operator!=(ManagedIterator const& rhs) const {
        return !(*this==rhs);
    }

    ManagedIterator& operator++() {
        ++mptr;
        return *this;
    }

    ManagedIterator operator++(int) {
        ManagedIterator tmp (*this);
        ++(*this);
        return tmp;
    }

    ManagedIterator& operator--() {
        --mptr;
        return *this;
    }

    ManagedIterator operator--(int) {
        ManagedIterator tmp (*this);
        --(*this);
        return tmp;
    }

    T *operator* () const {
        return mptr;
    }

private:
    T *mptr;
};

template<typename T>
struct ManagedPtr {
    typedef T type;
    typedef ManagedIterator<T> iterator;

    ManagedPtr(): msize(0), mptr(NULL) {
    }
    ManagedPtr(std::size_t n_elements): msize(n_elements * sizeof(T)) {
        if(n_elements)
            cudaMallocHost((void **)&mptr, msize);
    }
    ~ManagedPtr() {
        if(mptr)
            cudaFreeHost(mptr);
    }
    void reset(std::size_t n_elements) {
        if(n_elements * sizeof(T) == msize)
            return; //We dont neen to change anything;
        if(mptr) {
            cudaFreeHost(mptr);
            mptr = NULL;
        }
        if(n_elements) {
            msize = n_elements * sizeof(T);
            cudaMallocHost((void **)&mptr, msize);
        }
    }
    inline
    void clear() const {
        if(mptr && msize) {
            cudaMemset(mptr, 0x0, msize);
        }
    }
    inline
    T *operator()() {
        return (T*)mptr;
    }
    inline
    const T *operator()() const {
        return (T*)mptr;
    }

    inline
    T &operator[](std::size_t idx) {
        return ((T*)mptr)[idx];
    }
    inline
    const T &operator[](std::size_t idx) const {
        return ((T*)mptr)[idx];
    }
    inline
    std::size_t size() const {
        return msize/sizeof(T);
    }
    inline
    std::size_t bytes() const {
        return msize;
    }

    ManagedIterator<T> begin() const {
        return ManagedIterator<T>((T*)mptr);
    }
    ManagedIterator<T> end() const {
        return ManagedIterator<T>((T*)( reinterpret_cast<std::size_t>(mptr)+msize));
    }
private:
    void operator=(ManagedPtr<T> &p);
private:
    std::size_t msize;
    void *mptr;
};


#endif
