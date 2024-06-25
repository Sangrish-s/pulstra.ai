const { createProxyMiddleware } = require('http-proxy-middleware');

module.exports = (req, res) => {
  const proxy = createProxyMiddleware({
    target: 'http://184.72.131.74:8088',
    changeOrigin: true,
    pathRewrite: { '^/api': '' },
  });
  return proxy(req, res, (err) => {
    if (err) {
      res.status(500).send(err.message);
    }
  });
};
