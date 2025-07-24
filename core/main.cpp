#include <iostream>
#include <mujoco/mujoco.h>

int main()
{
  if (mjVERSION_HEADER != mj_version()) { std::cout << "Wrong Version Installed" << std::endl; }


  std::cout << "Success!" << std::endl;

  return 0;
}
