import random

u_l = 500
sigma_l = 50
u_r = 550
sigma_r = 100

value_l_init = 998
value_r_init = 998

value_l = []
value_r = []

def tiger_robot():
    # 得到0.1的概率
    rand = random.random()
    if rand < 0.1:
        if random.random() >= 0.5:
            value_l.append(random.uniform(u_l - sigma_l, u_l + sigma_l))
        else:
            value_r.append(random.uniform(u_r - sigma_r, u_r + sigma_r))
    else:
        if sum(value_l)/len(value_l) >= sum(value_r)/len(value_r):
            value_l.append(random.uniform(u_l - sigma_l, u_l + sigma_l))
        else:
            value_r.append(random.uniform(u_r - sigma_r, u_r + sigma_r))
if __name__ == '__main__':
    value_l.append(value_l_init)
    value_r.append(value_r_init)
    for i in range(500):
        tiger_robot()
    # print(value_l)
    # print(value_r)
    print(sum(value_l)/len(value_l))
    print(sum(value_r)/len(value_r))