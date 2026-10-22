// Package config parses the gateway's environment-only configuration.
package config

import (
	"errors"
	"fmt"
	"net/url"
	"os"
	"strconv"
	"strings"
	"time"
)

// Config is the complete runtime configuration. It is built once at startup
// and is immutable afterwards.
type Config struct {
	Addr         string
	APIKey       string
	UpstreamBase string
	Models       []string
	Model        string
	Proxy        string
	PoolMin      int
	PoolMax      int
	PoolTTL      time.Duration
	// MaxMessages 单次请求允许的最多消息条数。上游硬上限 100 条（实测
	// 100 通过、101 报 "This conversation is too long"），默认留 4 条余量。
	MaxMessages int
	// SessionPerRequest 每个上游请求都新取一个匿名 session token（不复用）。
	// 上游按 session 累积对话历史，复用过久会报 "This conversation is too long"。
	SessionPerRequest bool
	IdleTimeout  time.Duration
	// PreambleStyle selects the tool preamble skeleton ("hermes" or
	// "inception"); wired to tools.PreStyle at boot.
	PreambleStyle string
	// NativeWebSearch auto-declares a gateway-served web_search tool on
	// tool-bearing chat requests and executes its calls against the search
	// backend. Off by default.
	NativeWebSearch bool
	// ContextChars is a deprecated no-op, retained so existing deployments do
	// not fail to boot. There is no client-side length guard (SPEC §1.1).
	ContextChars   int
	SessionTimeout time.Duration
	// SearchTimeout is the /v1/web_search sliding idle watchdog.
	SearchTimeout time.Duration
	// SearchConfigCheck enables the one startup /api/search-lab/config probe.
	SearchConfigCheck bool
	// SearchProbeTimeout bounds that probe.
	SearchProbeTimeout time.Duration
	// AudioTimeout bounds the pre-header phase of one /api/tts or /api/stt call.
	AudioTimeout time.Duration
	// AudioMaxBytes caps a /v1/audio/transcriptions upload.
	AudioMaxBytes int64
	// DecisionTimeout bounds one whole /api/decide/run call.
	DecisionTimeout time.Duration
	// ClearanceRefresh is the proactive _vcrcs refresh age (cookie TTL 60m).
	ClearanceRefresh time.Duration
	// ClearanceCooldown is the minimum retry interval after a failed
	// clearance acquisition.
	ClearanceCooldown time.Duration
	// ClearancePool is the size of the clearance ring (SPEC §2.1.2).
	ClearancePool int
}

// Defaults returns the configuration used when no environment variable is set.
func Defaults() Config {
	return Config{
		Addr:            ":3000",
		UpstreamBase:    "https://chat.inceptionlabs.ai",
		Model:           "mercury-3",
		PoolMin:         2,
		PoolMax:         8,
		PoolTTL:         30 * time.Minute,
		MaxMessages:     96,
		IdleTimeout:     60 * time.Second,
		ContextChars:    0,
		SessionTimeout:  15 * time.Second,
		NativeWebSearch: false,

		SearchTimeout:      60 * time.Second,
		SearchConfigCheck:  true,
		SearchProbeTimeout: 15 * time.Second,

		AudioTimeout:    120 * time.Second,
		AudioMaxBytes:   25 << 20,
		DecisionTimeout: 120 * time.Second,

		ClearanceRefresh:  45 * time.Minute,
		ClearanceCooldown: 15 * time.Second,
		ClearancePool:     20,
	}
}

// Load reads IL2API_* variables from the process environment.
func Load() (Config, error) { return FromEnv(os.Getenv) }

// FromEnv parses configuration using getenv, which makes it testable without
// touching the process environment.
func FromEnv(getenv func(string) string) (Config, error) {
	c := Defaults()

	setString := func(key string, dst *string) {
		if v := strings.TrimSpace(getenv(key)); v != "" {
			*dst = v
		}
	}
	setString("IL2API_ADDR", &c.Addr)
	setString("IL2API_API_KEY", &c.APIKey)
	setString("IL2API_UPSTREAM_BASE", &c.UpstreamBase)
	setString("IL2API_MODEL", &c.Model)
	setString("IL2API_PREAMBLE_STYLE", &c.PreambleStyle)

	if v := strings.TrimSpace(getenv("IL2API_MODELS")); v != "" {
		for _, part := range strings.Split(v, ",") {
			if m := strings.TrimSpace(part); m != "" {
				c.Models = append(c.Models, m)
			}
		}
	}
	if len(c.Models) == 0 {
		c.Models = []string{c.Model}
	}

	proxy := strings.TrimSpace(getenv("IL2API_PROXY"))
	if proxy == "" {
		for _, key := range []string{"HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"} {
			if v := strings.TrimSpace(getenv(key)); v != "" {
				proxy = v
				break
			}
		}
	}
	proxy, err := normalizeProxy(proxy)
	if err != nil {
		return Config{}, err
	}
	c.Proxy = proxy

	if c.PoolMin, err = envInt(getenv, "IL2API_POOL_MIN", c.PoolMin); err != nil {
		return Config{}, err
	}
	if c.PoolMax, err = envInt(getenv, "IL2API_POOL_MAX", c.PoolMax); err != nil {
		return Config{}, err
	}
	if c.ContextChars, err = envInt(getenv, "IL2API_CONTEXT_CHARS", c.ContextChars); err != nil {
		return Config{}, err
	}
	if c.PoolTTL, err = envDuration(getenv, "IL2API_POOL_TTL", c.PoolTTL); err != nil {
		return Config{}, err
	}
	if c.MaxMessages, err = envInt(getenv, "IL2API_MAX_MESSAGES", c.MaxMessages); err != nil {
		return Config{}, err
	}
	if c.IdleTimeout, err = envDuration(getenv, "IL2API_IDLE_TIMEOUT", c.IdleTimeout); err != nil {
		return Config{}, err
	}
	if c.SessionTimeout, err = envDuration(getenv, "IL2API_SESSION_TIMEOUT", c.SessionTimeout); err != nil {
		return Config{}, err
	}
	if c.SearchTimeout, err = envDuration(getenv, "IL2API_SEARCH_TIMEOUT", c.SearchTimeout); err != nil {
		return Config{}, err
	}
	if c.SearchProbeTimeout, err = envDuration(getenv, "IL2API_SEARCH_PROBE_TIMEOUT", c.SearchProbeTimeout); err != nil {
		return Config{}, err
	}
	if c.AudioTimeout, err = envDuration(getenv, "IL2API_AUDIO_TIMEOUT", c.AudioTimeout); err != nil {
		return Config{}, err
	}
	if c.AudioMaxBytes, err = envInt64(getenv, "IL2API_AUDIO_MAX_BYTES", c.AudioMaxBytes); err != nil {
		return Config{}, err
	}
	if c.DecisionTimeout, err = envDuration(getenv, "IL2API_DECISIONS_TIMEOUT", c.DecisionTimeout); err != nil {
		return Config{}, err
	}
	if c.ClearanceRefresh, err = envDuration(getenv, "IL2API_CLEARANCE_REFRESH", c.ClearanceRefresh); err != nil {
		return Config{}, err
	}
	if c.ClearanceCooldown, err = envDuration(getenv, "IL2API_CLEARANCE_COOLDOWN", c.ClearanceCooldown); err != nil {
		return Config{}, err
	}
	if c.ClearancePool, err = envInt(getenv, "IL2API_CLEARANCE_POOL", c.ClearancePool); err != nil {
		return Config{}, err
	}
	if v := strings.TrimSpace(strings.ToLower(getenv("IL2API_SEARCH_CONFIG_CHECK"))); v != "" {
		switch v {
		case "on", "1", "true", "yes":
			c.SearchConfigCheck = true
		case "off", "0", "false", "no":
			c.SearchConfigCheck = false
		default:
			return Config{}, fmt.Errorf("config: IL2API_SEARCH_CONFIG_CHECK must be on or off, got %q", v)
		}
	}
	if v := strings.TrimSpace(strings.ToLower(getenv("IL2API_SESSION_PER_REQUEST"))); v != "" {
		switch v {
		case "on", "1", "true", "yes":
			c.SessionPerRequest = true
		case "off", "0", "false", "no":
			c.SessionPerRequest = false
		default:
			return Config{}, fmt.Errorf("config: IL2API_SESSION_PER_REQUEST must be on or off, got %q", v)
		}
	}
	if v := strings.TrimSpace(strings.ToLower(getenv("IL2API_NATIVE_WEB_SEARCH"))); v != "" {
		switch v {
		case "on", "1", "true":
			c.NativeWebSearch = true
		case "off", "0", "false":
			c.NativeWebSearch = false
		default:
			return Config{}, fmt.Errorf("config: IL2API_NATIVE_WEB_SEARCH must be on or off, got %q", v)
		}
	}

	if err := c.validate(); err != nil {
		return Config{}, err
	}
	return c, nil
}

func (c Config) validate() error {
	if c.PoolMin < 0 {
		return fmt.Errorf("config: IL2API_POOL_MIN must be >= 0, got %d", c.PoolMin)
	}
	if c.PoolMax < 1 {
		return fmt.Errorf("config: IL2API_POOL_MAX must be >= 1, got %d", c.PoolMax)
	}
	if c.PoolMin > c.PoolMax {
		return fmt.Errorf("config: IL2API_POOL_MIN (%d) must not exceed IL2API_POOL_MAX (%d)", c.PoolMin, c.PoolMax)
	}
	if c.MaxMessages < 4 {
		return fmt.Errorf("config: IL2API_MAX_MESSAGES must be >= 4, got %d", c.MaxMessages)
	}
	if c.PoolTTL <= 0 {
		return fmt.Errorf("config: IL2API_POOL_TTL must be positive, got %s", c.PoolTTL)
	}
	if c.IdleTimeout <= 0 {
		return fmt.Errorf("config: IL2API_IDLE_TIMEOUT must be positive, got %s", c.IdleTimeout)
	}
	if c.SessionTimeout <= 0 {
		return fmt.Errorf("config: IL2API_SESSION_TIMEOUT must be positive, got %s", c.SessionTimeout)
	}
	if c.SearchTimeout <= 0 {
		return fmt.Errorf("config: IL2API_SEARCH_TIMEOUT must be positive, got %s", c.SearchTimeout)
	}
	if c.SearchProbeTimeout <= 0 {
		return fmt.Errorf("config: IL2API_SEARCH_PROBE_TIMEOUT must be positive, got %s", c.SearchProbeTimeout)
	}
	if c.ContextChars < -1 {
		return fmt.Errorf("config: IL2API_CONTEXT_CHARS must be positive or -1, got %d", c.ContextChars)
	}
	if c.AudioTimeout <= 0 {
		return fmt.Errorf("config: IL2API_AUDIO_TIMEOUT must be positive, got %s", c.AudioTimeout)
	}
	if c.AudioMaxBytes <= 0 {
		return fmt.Errorf("config: IL2API_AUDIO_MAX_BYTES must be positive, got %d", c.AudioMaxBytes)
	}
	if c.DecisionTimeout <= 0 {
		return fmt.Errorf("config: IL2API_DECISIONS_TIMEOUT must be positive, got %s", c.DecisionTimeout)
	}
	if c.ClearanceRefresh <= 0 {
		return fmt.Errorf("config: IL2API_CLEARANCE_REFRESH must be positive, got %s", c.ClearanceRefresh)
	}
	if c.ClearanceCooldown < 0 {
		return fmt.Errorf("config: IL2API_CLEARANCE_COOLDOWN must not be negative, got %s", c.ClearanceCooldown)
	}
	if c.ClearancePool < 1 {
		return fmt.Errorf("config: IL2API_CLEARANCE_POOL must be >= 1, got %d", c.ClearancePool)
	}
	u, err := url.Parse(c.UpstreamBase)
	if err != nil || (u.Scheme != "http" && u.Scheme != "https") || u.Host == "" {
		return fmt.Errorf("config: IL2API_UPSTREAM_BASE must be an http(s) URL, got %q", c.UpstreamBase)
	}
	if !strings.HasPrefix(c.Addr, ":") && !strings.Contains(c.Addr, ":") {
		return fmt.Errorf("config: IL2API_ADDR must be host:port or :port, got %q", c.Addr)
	}
	return nil
}

// normalizeProxy accepts bare host:port, http://, and https:// proxy URLs and
// returns a canonical http(s) URL string. Empty input stays empty.
func normalizeProxy(raw string) (string, error) {
	if raw == "" {
		return "", nil
	}
	if !strings.Contains(raw, "://") {
		raw = "http://" + raw
	}
	u, err := url.Parse(raw)
	if err != nil {
		return "", fmt.Errorf("config: invalid proxy URL: %w", redactURLError(err, raw))
	}
	if u.Scheme != "http" && u.Scheme != "https" {
		return "", fmt.Errorf("config: proxy scheme must be http or https, got %q", u.Scheme)
	}
	if u.Host == "" {
		return "", fmt.Errorf("config: proxy %q has no host", RedactProxy(raw))
	}
	return u.String(), nil
}

// RedactProxy removes any userinfo (proxy credentials) from a proxy URL for
// logs and error messages.
func RedactProxy(raw string) string {
	u, err := url.Parse(raw)
	if err != nil {
		return "<invalid proxy URL>"
	}
	u.User = nil
	return u.String()
}

// redactURLError strips the raw string from a url.Parse error so a password
// embedded in userinfo never reaches the logs.
func redactURLError(err error, raw string) error {
	var ue *url.Error
	if errors.As(err, &ue) && ue.Err != nil {
		return ue.Err
	}
	return errors.New("malformed URL")
}

func envInt(getenv func(string) string, key string, def int) (int, error) {
	raw := strings.TrimSpace(getenv(key))
	if raw == "" {
		return def, nil
	}
	n, err := strconv.Atoi(raw)
	if err != nil {
		return 0, fmt.Errorf("config: %s must be an integer, got %q", key, raw)
	}
	return n, nil
}

func envInt64(getenv func(string) string, key string, def int64) (int64, error) {
	raw := strings.TrimSpace(getenv(key))
	if raw == "" {
		return def, nil
	}
	n, err := strconv.ParseInt(raw, 10, 64)
	if err != nil {
		return 0, fmt.Errorf("config: %s must be an integer, got %q", key, raw)
	}
	return n, nil
}

func envDuration(getenv func(string) string, key string, def time.Duration) (time.Duration, error) {
	raw := strings.TrimSpace(getenv(key))
	if raw == "" {
		return def, nil
	}
	d, err := time.ParseDuration(raw)
	if err != nil {
		return 0, fmt.Errorf("config: %s must be a duration such as 30m or 60s, got %q", key, raw)
	}
	return d, nil
}
