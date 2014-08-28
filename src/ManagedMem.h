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

    ManagedPtr(): sz(0), ptr(NULL) {
    }
    ManagedPtr(std::size_t n_bytes): sz(n_bytes) {
        cudaMallocHost((void **)&ptr, n_bytes);
    }
    ~ManagedPtr() {
        if(ptr)
            cudaFreeHost(ptr);
    }
    inline
    T *operator()() {
        return (T*)ptr;
    }
    inline
    const T *operator()() const {
        return (T*)ptr;
    }

    inline
    T &operator[](std::size_t idx) {
        return ((T*)ptr)[idx];
    }
    inline
    const T &operator[](std::size_t idx) const {
        return ((T*)ptr)[idx];
    }

    inline
    std::size_t size() const {
        return sz/sizeof(T);
    }
    inline
    std::size_t bytes() const {
        return sz;
    }

    ManagedIterator<T> begin() const {
        return ManagedIterator<T>((T*)ptr);
    }
    ManagedIterator<T> end() const {
        return ManagedIterator<T>((T*)( reinterpret_cast<std::size_t>(ptr)+sz));
    }

private:
    std::size_t sz;
    void *ptr;
};


#endif
