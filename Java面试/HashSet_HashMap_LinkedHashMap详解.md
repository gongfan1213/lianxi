# HashSet、HashMap、LinkedHashMap详解（面试话术版）

---

## 1. HashSet

### 1.1 基本介绍
**面试话术：**
HashSet是Java中常用的无序、不可重复集合，底层其实是基于HashMap实现的。它主要用于去重和快速查找。

### 1.2 底层原理
- HashSet底层维护了一个HashMap，每个加入HashSet的元素其实是作为HashMap的key，value是一个固定的Object对象（private static final Object PRESENT = new Object();）。
- 添加元素时，先计算元素的hashCode，然后通过哈希算法定位到数组的某个位置，再通过equals判断是否重复。
- 不允许重复元素，判断依据是hashCode和equals。

### 1.3 常见用法
```java
Set<String> set = new HashSet<>();
set.add("A");
set.add("B");
set.add("A"); // 不会重复
```

### 1.4 注意点
- 元素必须重写hashCode和equals方法，否则去重不生效。
- 线程不安全，多线程场景用Collections.synchronizedSet或ConcurrentHashMap的keySet。
- 无序，遍历顺序不固定。

---

## 2. HashMap

### 2.1 基本介绍
**面试话术：**
HashMap是Java中最常用的键值对（key-value）映射容器，允许key和value为null，查询和插入效率高，适合存储大量数据。

### 2.2 底层原理（JDK1.8为例）
- 底层结构是"数组+链表+红黑树"。
- 初始是一个Node数组，key经过hash运算定位到数组的某个桶（bucket）。
- 桶内如果有哈希冲突（不同key算到同一个桶），用链表存储；链表长度超过8时转为红黑树，提高查询效率。
- put时先判断hash和equals，key已存在则覆盖，否则插入。
- 扩容：当元素数量超过阈值（默认0.75*容量）时，数组容量翻倍，所有元素重新分布。

### 2.3 常见用法
```java
Map<String, Integer> map = new HashMap<>();
map.put("A", 1);
map.put("B", 2);
map.put("A", 3); // 覆盖原有key
System.out.println(map.get("A")); // 输出3
```

### 2.4 注意点
- key必须重写hashCode和equals，否则会出现查找异常。
- 线程不安全，多线程用ConcurrentHashMap。
- key无序，遍历顺序不固定。
- 容量和负载因子影响性能和内存占用。

---

## 3. LinkedHashMap

### 3.1 基本介绍
**面试话术：**
LinkedHashMap是HashMap的子类，底层在HashMap的基础上增加了一条双向链表，记录元素的插入顺序或访问顺序。

### 3.2 底层原理
- 继承自HashMap，除了哈希表结构外，每个节点还维护了before/after指针，形成双向链表。
- 默认按插入顺序遍历（也可设置为访问顺序，常用于LRU缓存）。
- put、get等操作会维护链表顺序。

### 3.3 常见用法
```java
Map<String, Integer> map = new LinkedHashMap<>();
map.put("A", 1);
map.put("B", 2);
map.put("C", 3);
for (String key : map.keySet()) {
    System.out.print(key + " "); // 输出A B C
}
```

### 3.4 注意点
- 有序，默认插入顺序，也可按访问顺序。
- 性能略低于HashMap，但有序性很有用。
- 适合实现LRU缓存（重写removeEldestEntry方法）。

---

## 4. 三者区别总结
| 特性         | HashSet         | HashMap         | LinkedHashMap         |
|--------------|-----------------|-----------------|----------------------|
| 存储结构     | HashMap实现     | 数组+链表+树    | HashMap+双向链表     |
| 是否有序     | 无序            | 无序            | 有序（插入/访问）    |
| 是否允许null | 允许            | 允许            | 允许                 |
| 线程安全     | 否              | 否              | 否                   |
| 典型应用     | 去重集合        | 键值对存储      | 有序键值对/LRU缓存   |

---

## 5. 面试高频问题
1. HashSet为什么不能存放重复元素？
   - 因为底层用HashMap的key存储元素，key不能重复。
2. HashMap和Hashtable区别？
   - HashMap线程不安全，效率高，允许null；Hashtable线程安全，效率低，不允许null。
3. LinkedHashMap如何实现有序？
   - 通过维护一条双向链表，记录插入或访问顺序。
4. HashMap为什么要扩容？
   - 保证查询和插入效率，避免哈希冲突过多导致链表/树过长。

---

> 以上为HashSet、HashMap、LinkedHashMap的通俗易懂详细讲解，适合面试和日常复习。 