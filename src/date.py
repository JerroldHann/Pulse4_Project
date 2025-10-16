from datetime import datetime
# 假设 744 对应的时间为 2025年10月16日 19:00:00
base_time = datetime(2025, 10, 16, 19, 0, 0)
max_step_database = 744
def date_to_step_range(start_date, end_date):
    """
    Convert start and end dates to step range (inclusive), 
    where step = hours since 2025-01-01 00:00:00.
    
    :param start_date: Start datetime (YYYY-MM-DD HH:MM:SS)
    :param end_date: End datetime (YYYY-MM-DD HH:MM:SS)
    :return: step range (start_step, end_step)
    """
    
    # 将输入的日期字符串转换为 datetime 对象
    start_dt = datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S")
    
    # 计算从基准时间到目标时间的时间差（单位为小时）
    start_diff = base_time - start_dt
    end_diff = base_time - end_dt
    
    # 计算起始和结束时间的 step 值（每小时一个 step）
    start_step = max_step_database - int(start_diff.total_seconds() // 3600)  # 转换为小时数
    end_step = max_step_database -int(end_diff.total_seconds() // 3600)  # 转换为小时数
   
    return start_step, end_step

def date_to_step(date_str):
    """
    Convert a date string (YYYY-MM-DD HH:MM:SS) to step (hours since 2025-01-01 00:00:00).
    """
    target_time = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
    delta = target_time - base_time
    step = int(delta.total_seconds() // 3600)  + max_step_database # Convert to hours
    return step


