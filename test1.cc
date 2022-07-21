#include <iostream>
#include <vector>

void alg(std::vector<int>& nums){
	if (nums.size()==0)
        return;
    int p = nums.size();
    int q = p;
    while(p > 0){
        if (nums[p] != 0){
            int tmp = nums[p];
            nums[p] = nums[q];
            nums[q] = tmp;
            q--;
        }
        p--;
    }

}

int main(){
    std::vector<int> nums{0, 1, 2, 3, 0, 0, 4, 0, 5};
    alg(nums);
    for (auto i: nums){
        std::cout << i << ' ';
    }
    std::cout << std::endl;
}