import { useEffect, useMemo, useState } from "react";
import { Button, Form, Input, InputNumber, Typography } from "antd";

const { Title, Text } = Typography;

type StageKey = "FINAM" | "AI" | "TRADE" | "FORM";

export type LandingFormValues = {
  startDate: string;
  endDate: string;
  hotNewsCount: number;
};

const STAGES: StageKey[] = ["FINAM", "AI", "TRADE", "FORM"];
const TEXT_FADE_OUT_DELAY = 2000;
const TEXT_STAGE_SWITCH_DELAY = 2400;
const EXIT_DELAY = 900;

const parseDate = (value?: string): Date | null => {
  if (!value) {
    return null;
  }
  const match = value.trim().match(/^(\d{2})\.(\d{2})\.(\d{4})$/);
  if (!match) {
    return null;
  }
  const day = Number(match[1]);
  const month = Number(match[2]);
  const year = Number(match[3]);
  const parsed = new Date(year, month - 1, day);
  if (
    parsed.getFullYear() !== year ||
    parsed.getMonth() !== month - 1 ||
    parsed.getDate() !== day
  ) {
    return null;
  }
  return parsed;
};

type LandingScreenProps = {
  onComplete: (values: LandingFormValues) => void;
  loading?: boolean;
};

const LandingScreen = ({ onComplete, loading = false }: LandingScreenProps) => {
  const [stageIndex, setStageIndex] = useState<number>(0);
  const [textVisible, setTextVisible] = useState<boolean>(true);
  const [exiting, setExiting] = useState<boolean>(false);
  const [form] = Form.useForm<LandingFormValues>();

  const stage = STAGES[stageIndex];

  useEffect(() => {
    if (stage === "FORM") {
      setTextVisible(true);
      return;
    }

    setTextVisible(true);

    const fadeTimer = window.setTimeout(() => setTextVisible(false), TEXT_FADE_OUT_DELAY);
    const nextTimer = window.setTimeout(() => {
      setStageIndex((prev) => Math.min(prev + 1, STAGES.length - 1));
    }, TEXT_STAGE_SWITCH_DELAY);

    return () => {
      window.clearTimeout(fadeTimer);
      window.clearTimeout(nextTimer);
    };
  }, [stage]);

  const headline = useMemo(() => {
    switch (stage) {
      case "FINAM":
        return "FINAM x HSE";
      case "AI":
        return "AI";
      case "TRADE":
        return "TRADE HACK";
      default:
        return "";
    }
  }, [stage]);

  const handleFinish = (values: LandingFormValues) => {
    if (loading) {
      return;
    }
    setExiting(true);
    window.setTimeout(() => {
      onComplete(values);
    }, EXIT_DELAY);
  };

  return (
    <div className={`landing-overlay${exiting ? " landing-overlay--hidden" : ""}`}>
      <div className={`landing-card${stage === "FORM" ? " landing-card--form" : ""}`}>
        {stage !== "FORM" ? (
          <Title level={1} className={`landing-title${textVisible ? " landing-title--visible" : ""}`}>
            {headline}
          </Title>
        ) : (
          <>
            <Title level={2} className="landing-form-title">
              Настройте поиск новостей
            </Title>
            <Text className="landing-form-subtitle">
              Используйте формат даты ДД.ММ.ГГГГ, например 01.09.2024
            </Text>
            <Form<LandingFormValues>
              form={form}
              layout="vertical"
              className="landing-form"
              onFinish={handleFinish}
              initialValues={{ hotNewsCount: 3 }}
              requiredMark={false}
              disabled={loading}
            >
              <Form.Item
                label="Введите дату начала поиска новостей"
                name="startDate"
                rules={[
                  { required: true, message: "Введите дату начала" },
                  {
                    validator: (_, value) => {
                      if (!value) {
                        return Promise.resolve();
                      }
                      return parseDate(value)
                        ? Promise.resolve()
                        : Promise.reject(new Error("Формат ДД.ММ.ГГГГ"));
                    },
                  },
                ]}
              >
                <Input placeholder="Например: 01.09.2024" size="large" />
              </Form.Item>
              <Form.Item
                label="Введите дату конца поиска новостей"
                name="endDate"
                dependencies={["startDate"]}
                rules={[
                  { required: true, message: "Введите дату окончания" },
                  ({ getFieldValue }) => ({
                    validator: (_, value) => {
                      if (!value) {
                        return Promise.resolve();
                      }
                      const parsedEnd = parseDate(value);
                      if (!parsedEnd) {
                        return Promise.reject(new Error("Формат ДД.ММ.ГГГГ"));
                      }
                      const parsedStart = parseDate(getFieldValue("startDate"));
                      if (!parsedStart) {
                        return Promise.resolve();
                      }
                      if (parsedEnd < parsedStart) {
                        return Promise.reject(new Error("Дата окончания не может быть раньше начала"));
                      }
                      return Promise.resolve();
                    },
                  }),
                ]}
              >
                <Input placeholder="Например: 30.09.2024" size="large" />
              </Form.Item>
              <Form.Item
                label="Количество горячих новостей"
                name="hotNewsCount"
                rules={[
                  { required: true, message: "Введите количество" },
                  {
                    type: "number",
                    min: 1,
                    message: "Минимум 1 новость",
                  },
                ]}
              >
                <InputNumber
                  min={1}
                  placeholder="Например: 5"
                  size="large"
                  className="landing-number-input"
                  style={{ width: "100%" }}
                />
              </Form.Item>
              <Form.Item className="landing-submit">
                <Button type="primary" htmlType="submit" size="large" loading={loading} disabled={loading}>
                  Перейти к приложению
                </Button>
              </Form.Item>
            </Form>
          </>
        )}
      </div>
    </div>
  );
};

export default LandingScreen;
