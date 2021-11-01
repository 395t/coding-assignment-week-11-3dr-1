SCRIPT=`realpath $0`
SCRIPTPATH=`dirname $SCRIPT`

cd $SCRIPTPATH/..
wget https://shapenet.cs.stanford.edu/ericyi/shapenetcore_partanno_segmentation_benchmark_v0.zip --no-check-certificate
unzip shapenetcore_partanno_segmentation_benchmark_v0.zip
rm shapenetcore_partanno_segmentation_benchmark_v0.zip

wget http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip --no-check-certificate
unzip ModelNet10.zip
rm ModelNet10.zip

wget http://modelnet.cs.princeton.edu/ModelNet40.zip --no-check-certificate
unzip ModelNet40.zip
rm ModelNet40.zip

cd -
