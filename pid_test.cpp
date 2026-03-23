#include <iostream>

// 定义一个简单的 PID 控制器类
class PIDController {
private:
    double kp, ki, kd;
    double integral, previous_error;

public:
    // 构造函数：初始化 PID 参数
    PIDController(double p, double i, double d) : kp(p), ki(i), kd(d), integral(0), previous_error(0) {}

    // 计算控制信号
    double compute(double setpoint, double measured_value) {
        double error = setpoint - measured_value;
        integral += error;
        double derivative = error - previous_error;
        
        // 核心公式：u(t) = Kp*e(t) + Ki*∫e(t)dt + Kd*de(t)/dt
        double output = kp * error + ki * integral + kd * derivative;
        
        previous_error = error;
        return output;
    }
};

int main() {
    std::cout << "🚀 C++ 编译环境就绪！PID 控制系统启动..." << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    // 实例化一个 PID 控制器，设置 Kp=0.1, Ki=0.01, Kd=0.05
    PIDController myPID(0.1, 0.01, 0.05);
    
    double current_position = 0.0; // 初始位置
    double target_position = 100.0; // 目标位置 (比如控制电机转到 100 度)
    
    std::cout << "目标位置: " << target_position << ", 当前位置: " << current_position << "\n\n";

    // 模拟 5 次控制循环
    for (int i = 1; i <= 5; ++i) {
        // 1. 计算控制量 (比如输出给电机的电压)
        double control_signal = myPID.compute(target_position, current_position);
        
        // 2. 模拟系统响应 (假设系统 1:1 响应控制信号，实际物理系统会复杂得多)
        current_position += control_signal; 
        
        std::cout << "第 " << i << " 次迭代 -> 控制输出(u): " << control_signal 
                  << " | 传感器读数(y): " << current_position << std::endl;
    }

    return 0;
}