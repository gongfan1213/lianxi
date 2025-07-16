# volatile通俗易懂讲解（面试话术版）

---

## 1. volatile是什么？
**面试话术：**
volatile是Java中的一个轻量级同步关键字，用来修饰变量。它保证多线程环境下变量的可见性和有序性，但不保证原子性。

---

## 2. volatile的作用
- **可见性**：一个线程修改了volatile变量，其他线程能立刻看到最新值。
- **禁止指令重排序**：volatile变量的读写前后，JVM会插入内存屏障，防止代码重排序，保证执行顺序。

---

## 3. 底层原理
**面试话术：**
- 每个线程有自己的工作内存（缓存），普通变量的修改不会立刻同步到主内存。
- volatile变量写操作会强制把新值刷新到主内存，读操作会强制从主内存读取最新值。
- JVM通过插入内存屏障（Memory Barrier）实现可见性和有序性。
- 在CPU层面，volatile变量会加上lock前缀的汇编指令，保证多核下的可见性。

---

## 4. volatile和synchronized的区别
| 特性         | volatile           | synchronized         |
|--------------|--------------------|---------------------|
| 可见性       | 有                 | 有                  |
| 原子性       | 无                 | 有                  |
| 有序性       | 有（部分）         | 有                  |
| 性能         | 高                 | 低（有锁开销）      |
| 用法         | 修饰变量           | 修饰代码块/方法      |

---

## 5. 典型用法
1. 状态标志：
```java
volatile boolean running = true;
while (running) {
    // do something
}
```
2. 单例模式中的双重检查锁（DCL）：
```java
public class Singleton {
    private static volatile Singleton instance;
    // ...
}
```

---

## 6. 注意点
- 只能保证可见性和有序性，不能保证复合操作的原子性（如i++）。
- 不适合计数器、累加器等需要原子操作的场景。
- 适合用作状态标志、开关量等简单场景。
- 多线程下的安全性更高的需求，建议用synchronized或原子类（AtomicInteger等）。

---

## 7. 面试高频问题
1. volatile能保证原子性吗？
   - 不能，只保证可见性和有序性。
2. volatile和synchronized的区别？
   - volatile不加锁，性能高，但不保证原子性；synchronized加锁，保证原子性和可见性。
3. volatile适合哪些场景？
   - 状态标志、单例DCL、简单的信号量。
4. 为什么DCL要用volatile？
   - 防止指令重排序，保证对象初始化安全。

---

> 以上为volatile的通俗易懂详细讲解，适合面试和日常复习。 