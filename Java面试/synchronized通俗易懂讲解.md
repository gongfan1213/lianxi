# synchronized通俗易懂讲解（面试话术版）

---

## 1. synchronized是什么？
**面试话术：**
synchronized是Java中最常用的线程同步关键字，用来保证多线程环境下代码的安全性。它可以修饰方法或代码块，确保同一时刻只有一个线程能执行被synchronized保护的代码。

---

## 2. 基本用法
- 修饰实例方法：锁住当前对象（this）
- 修饰静态方法：锁住当前类的Class对象
- 修饰代码块：可以指定锁对象

**示例：**
```java
public synchronized void method1() { /* ... */ } // 锁this
public static synchronized void method2() { /* ... */ } // 锁Class对象
public void method3() {
    synchronized(this) { /* ... */ } // 锁this
    synchronized(SomeClass.class) { /* ... */ } // 锁Class对象
    Object lock = new Object();
    synchronized(lock) { /* ... */ } // 锁自定义对象
}
```

---

## 3. 底层原理
**面试话术：**
synchronized的底层是JVM的Monitor（监视器锁）机制。每个对象都有一个Monitor，synchronized加锁时会尝试获取对象的Monitor，只有获取到锁的线程才能进入同步代码。

- JVM通过对象头中的Mark Word记录锁的状态。
- 支持无锁、偏向锁、轻量级锁、重量级锁四种状态，按需升级。
- synchronized代码块在字节码层面会插入`monitorenter`和`monitorexit`指令。

---

## 4. 锁的升级过程
**面试话术：**
JVM为了提升性能，引入了锁的升级机制：
- **无锁**：对象未被任何线程加锁。
- **偏向锁**：只有一个线程访问时，直接在对象头记录线程ID，避免CAS操作。
- **轻量级锁**：多个线程交替访问时，采用CAS自旋，不阻塞线程。
- **重量级锁**：多个线程同时竞争时，线程阻塞挂起，操作系统层面的互斥量，开销最大。

锁会根据竞争情况自动升级，减少性能损耗。

---

## 5. 常见面试问题
1. synchronized和Lock的区别？
   - synchronized是JVM层面实现，Lock是Java代码实现，Lock更灵活（可中断、公平锁、读写锁等）。
2. synchronized加在静态方法和实例方法上的区别？
   - 静态方法锁Class对象，实例方法锁this。
3. synchronized如何实现可重入？
   - 同一线程多次获得同一把锁不会死锁，JVM会记录加锁次数。
4. synchronized会发生死锁吗？
   - 会，如果多个线程互相等待对方持有的锁。

---

## 6. 注意点
- synchronized保证原子性和可见性，但不保证有序性。
- 不建议锁字符串常量或全局对象，容易引发死锁或性能问题。
- synchronized块尽量缩小范围，减少锁竞争。
- JDK1.6后synchronized性能大幅提升，已能满足大多数场景。

---

> 以上为synchronized的通俗易懂详细讲解，适合面试和日常复习。 