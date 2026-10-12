set -e
export PATH=$HOME/node22/bin:$PATH
TIL=/Users/macplayer1/ahnaf/tilawa
B=~/cpu-runner; rm -rf $B; mkdir -p $B/bundle
# 1. SDK build (arm64 native)
cp -r $TIL/packages/core $B/core && rm -rf $B/core/node_modules $B/core/dist
cd $B/core && npm ci --no-audit --no-fund >/tmp/core-ci.log 2>&1 && npm run build >>/tmp/core-ci.log 2>&1
echo "core built: $(ls dist | head -3 | tr "\n" " ")"
# 2. onnxruntime-node for arm64 linux
cd $B && npm init -y >/dev/null && npm install --no-audit --no-fund onnxruntime-node@1.21.0 >/tmp/ort.log 2>&1
echo "ort installed: $(node -e "require(\"$B/node_modules/onnxruntime-node\"); console.log(\"ok\")")"
# 3. node 22 arm64 binary
V=$(curl -s https://nodejs.org/dist/latest-v22.x/SHASUMS256.txt | grep linux-arm64.tar.xz | awk "{print \$2}" | sed "s/node-//;s/-linux-arm64.tar.xz//" | head -1)
curl -sSLO https://nodejs.org/dist/latest-v22.x/node-$V-linux-arm64.tar.xz
echo "node $V fetched"
