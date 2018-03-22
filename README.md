# 天池比赛介绍
[印象盐城·数创未来大数据竞赛 - 盐城汽车上牌量预测](https://tianchi.aliyun.com/competition/introduction.htm?spm=5176.100067.5678.1.3ce53cddlnx70h&raceId=231641)

# 比赛成绩
初赛45, 复赛17, 共2635只队

# 解决思路
1. 还原成真实日期：初赛date mapping区间是2013-01-02到2017-11-28, 复赛date mapping区间是2013-01-01到2017-11-28;
2. 参考2013年到2017年的放假通知将日期赋予以下类型(date_type): 国家规定周末上班日, 国家规定假期加班日, 正常工作日(星期一至星期五),正常周末加班日(星期六星期天), 国家规定假期后的第1个上班日
3. 使用brand, date_type,  day_of_week, year, month, week_of_year(1-52)作为特征, 其中brand, date_type, day_of_week用做类别变量。
4. 随机选取0.1作为验证集，仅用GBR算法的结果提交。
