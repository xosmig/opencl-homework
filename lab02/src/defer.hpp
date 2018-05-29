#ifndef OPENCLLAB01_DEFER_HPP
#define OPENCLLAB01_DEFER_HPP

#include <memory>

typedef std::shared_ptr<void> Defer;

template<typename F>
Defer defer(F f) {
  return Defer(nullptr, [f{std::move(f)}](...) { f(); });
}


#endif  // OPENCLLAB01_DEFER_HPP
