from flytekit import task, ImageSpec, Resources
from flytekit.extras.accelerators import L4


image = ImageSpec(
    requirements="pyproject.toml",
    name="liger-kernel",
    apt_packages=["build-essential", "git"],
)


@task(container_image=image, requests=Resources(gpu="1", mem="4Gi"), accelerator=L4)
def liger_tests():
    import subprocess
    subprocess.run(["pwd"], check=True)
    subprocess.run(["ls", "-l"], check=True)
    subprocess.run(["pip", "install", "-e", ".[dev]"], check=True)
    subprocess.run(["make", "test"], check=True)
    subprocess.run(["make", "test-convergence"], check=True)


@task(container_image=image.with_packages("transformers==4.44.2"), requests=Resources(gpu="1", mem="4Gi"), accelerator=L4)
def liger_tests_bwd():
    import subprocess

    subprocess.run(["pwd"], check=True)
    subprocess.run(["ls", "-l"], check=True)
    subprocess.run(["pip", "install", "-e", ".[dev]"], check=True)
    subprocess.run(["make", "test"], check=True)
    subprocess.run(["make", "test-convergence"], check=True)
